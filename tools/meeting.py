#!/usr/bin/env python3
"""Прототип: запись с диктофона AIREC → текст по спикерам → протокол → PDF.

Цепочка:
  1. .ops/.wav → PCM 16 кГц моно (tools/ops2wav.py)
  2. Whisper (mlx-whisper, large-v3-turbo) → текст с таймкодами слов
  3. Деление по спикерам (sherpa-onnx: pyannote segmentation 3.0 + WeSpeaker ResNet34)
  4. ИИ (Qwen) → резюме, темы, договорённости, задачи, контекст (JSON)
     - LLM_BASE_URL задан → любой OpenAI-совместимый API (рабочий Qwen и т.п.)
     - иначе → локальная модель через mlx-lm (LOCAL_LLM, по умолчанию Qwen3.5-9B-4bit)
  5. PDF по шаблону docs/SUMMARY_TEMPLATE.md

Использование:
    python tools/meeting.py captures/ops/20261121175353.ops
    python tools/meeting.py --recorder            # забрать новые записи с подключённого диктофона
    python tools/meeting.py FILE --speakers 2     # если число собеседников известно
    python tools/meeting.py FILE --no-llm         # только текст по спикерам, без протокола

Результат: meetings/<дата_время>/{transcript.json, protocol.json, <дата_время>.pdf}
Промежуточные файлы кэшируются: повторный запуск не расшифровывает заново (--redo, чтобы пересчитать).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import urllib.request
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ops2wav import decode_ops  # noqa: E402

SR = 16000
RECORDER_DIR = Path("/Volumes/NO NAME/RECORD")
OPS_DIR = ROOT / "captures" / "ops"
OUT_DIR = ROOT / "meetings"
# Модели лежат в одной папке, которую видно в Finder (по умолчанию ~/ИИ, ярлык на рабочем столе).
# Если папки нет — берём модели из Hugging Face (скачаются в кэш при первом запуске).
AI_DIR = Path(os.environ.get("AI_MODELS_DIR", Path.home() / "ИИ"))


def _model(local: str, hf_repo: str) -> str:
    p = AI_DIR / local
    return str(p) if p.is_dir() else hf_repo


WHISPER_MODEL = _model("Whisper (расшифровка)", "mlx-community/whisper-large-v3-turbo")
LOCAL_LLM = os.environ.get("LOCAL_LLM") or _model("Qwen 3.5 9B (протоколы)", "mlx-community/Qwen3.5-9B-4bit")
DIAR_SEG = AI_DIR / "Спикеры (деление по голосам)" / "segmentation.onnx"
DIAR_EMB = AI_DIR / "Спикеры (деление по голосам)" / "voice-embedding.onnx"


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def load_env() -> None:
    """Читает .env в корне проекта (KEY=VALUE), не перетирая уже заданные переменные."""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ---------- 1. Аудио ----------

def load_audio(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".ops":
        pcm = decode_ops(path.read_bytes())
    else:
        with wave.open(str(path)) as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SR, 1, 2):
                sys.exit(f"{path}: нужен WAV 16 кГц моно 16 бит")
            pcm = w.readframes(w.getnframes())
    return np.frombuffer(pcm, np.int16).astype(np.float32) / 32768


def recorded_at(path: Path) -> datetime:
    m = re.match(r"(\d{14})", path.stem)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    return datetime.fromtimestamp(path.stat().st_mtime)


def import_from_recorder() -> list[Path]:
    if not RECORDER_DIR.is_dir():
        sys.exit("Диктофон не подключён (нет /Volumes/NO NAME/RECORD)")
    OPS_DIR.mkdir(parents=True, exist_ok=True)
    new = []
    for f in sorted(RECORDER_DIR.glob("*.ops")):
        dst = OPS_DIR / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
            new.append(dst)
    log(f"с диктофона скопировано новых записей: {len(new)}")
    return new


# ---------- 2. Расшифровка ----------

def transcribe(audio: np.ndarray) -> list[dict]:
    """Слова с таймкодами: [{start, end, word}]."""
    import mlx_whisper

    r = mlx_whisper.transcribe(
        audio, path_or_hf_repo=WHISPER_MODEL, language="ru",
        condition_on_previous_text=False, temperature=0.0,
        word_timestamps=True, hallucination_silence_threshold=2.0,
    )
    words = []
    for seg in r["segments"]:
        for w in seg.get("words", []):
            if w["word"].strip():
                words.append({"start": float(w["start"]), "end": float(w["end"]), "word": w["word"]})
    return words


# ---------- 3. Спикеры ----------

SHERPA_REL = "https://github.com/k2-fsa/sherpa-onnx/releases/download"


def download_diar_models() -> None:
    """Скачивает модели деления по спикерам (~31 МБ) в папку моделей при первом запуске."""
    import io
    import tarfile

    DIAR_SEG.parent.mkdir(parents=True, exist_ok=True)
    log(f"скачиваю модели деления по спикерам в {DIAR_SEG.parent}")
    if not DIAR_SEG.exists():
        url = f"{SHERPA_REL}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
        with urllib.request.urlopen(url) as r, tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:bz2") as tar:
            member = tar.getmember("sherpa-onnx-pyannote-segmentation-3-0/model.onnx")
            DIAR_SEG.write_bytes(tar.extractfile(member).read())
    if not DIAR_EMB.exists():
        url = f"{SHERPA_REL}/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx"
        with urllib.request.urlopen(url) as r:
            DIAR_EMB.write_bytes(r.read())


def diarize(audio: np.ndarray, num_speakers: int | None) -> list[dict]:
    """Отрезки речи с номером спикера: [{start, end, speaker}] (speaker с 0)."""
    import sherpa_onnx

    seg, emb = DIAR_SEG, DIAR_EMB
    if not seg.exists() or not emb.exists():
        download_diar_models()
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(seg)),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb)),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers or -1, threshold=0.7),
        min_duration_on=0.3, min_duration_off=0.5,
    )
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    res = sd.process(audio).sort_by_start_time()
    return [{"start": s.start, "end": s.end, "speaker": s.speaker} for s in res]


def merge(words: list[dict], turns: list[dict], gap: float = 1.5) -> list[dict]:
    """Слова → реплики: у каждого слова спикер с наибольшим перекрытием (или ближайший отрезок)."""
    def speaker_of(w):
        best, best_ov = None, 0.0
        for t in turns:
            ov = min(w["end"], t["end"]) - max(w["start"], t["start"])
            if ov > best_ov:
                best, best_ov = t["speaker"], ov
        if best is None and turns:
            mid = (w["start"] + w["end"]) / 2
            best = min(turns, key=lambda t: min(abs(mid - t["start"]), abs(mid - t["end"])))["speaker"]
        return best or 0

    # Перенумеровать спикеров по порядку появления: 1, 2, 3…
    order: dict[int, int] = {}
    utts: list[dict] = []
    for w in words:
        sp = speaker_of(w)
        sp = order.setdefault(sp, len(order) + 1)
        last = utts[-1] if utts else None
        # Длинный монолог режем на абзацы по концу предложения, чтобы у каждого был свой таймкод.
        # Если Whisper не расставил точки — режем по паузе между словами, а совсем длинные абзацы — принудительно.
        if last:
            dur, pause = last["end"] - last["start"], w["start"] - last["end"]
            sentence_break = (last["text"].rstrip()[-1:] in ".?!…" and dur > 8) or \
                (dur > 12 and pause > 0.1) or dur > 20
        else:
            sentence_break = False
        if last and last["speaker"] == sp and w["start"] - last["end"] < gap and not sentence_break:
            last["text"] += w["word"]
            last["end"] = w["end"]
        else:
            utts.append({"start": w["start"], "end": w["end"], "speaker": sp, "text": w["word"]})
    for u in utts:
        u["text"] = u["text"].strip()
    return utts


# ---------- 4. Протокол (ИИ) ----------

PROMPT = """Ты — секретарь, который пишет протоколы встреч на русском языке.
Ниже расшифровка аудиозаписи с таймкодами [мм:сс] и спикерами. Расшифровка автоматическая и может содержать ошибки распознавания — исправляй очевидные, но не выдумывай фактов, которых нет в тексте.

Верни ТОЛЬКО JSON без пояснений и без markdown, строго такой структуры:
{
  "title": "короткое название встречи, 3-6 слов",
  "summary": "краткое резюме, 3-7 предложений о сути обсуждения",
  "topics": [{"title": "тема", "summary": "краткое содержание", "timecodes": "мм:сс–мм:сс"}],
  "agreements": [{"title": "договорённость или решение", "summary": "краткое содержание"}],
  "tasks": [{"task": "что сделать", "owner": "кто (имя) или —", "deadline": "срок или —", "timecode": "мм:сс"}],
  "context": "3-7 предложений о важных деталях, рисках, смысловых акцентах",
  "speakers_count": число_собеседников,
  "quality_comment": "качество записи: шумы, перебивания, неразборчивые места — по признакам в расшифровке"
}
Правила:
- Договорённости — то, что решили или согласовали. Задачи — конкретные действия, которые кто-то должен сделать.
- Если ответственный или срок не прозвучали — пиши "—", не выдумывай.
- Ответственный — человек, названный в тексте рядом с задачей («Сергей поговорит…» → Сергей), даже если имя
  написано с маленькой буквы или искажено распознаванием. «Спикер N» пиши, только если имени нет совсем.
- Если человек в записи один и пересказывает встречу — имена из его рассказа и есть ответственные.
- Не добавляй в задачи деталей, которых нет в тексте.
- В комментарии о качестве отметь, если в расшифровке видны ошибки распознавания (искажённые слова,
  оборванные фразы, нет начала записи).
- Пустые разделы — пустой список [].

Дата записи: {date}
Расшифровка:
{transcript}
"""


def fmt_tc(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 60:02d}:{sec % 60:02d}"


def transcript_text(utts: list[dict]) -> str:
    return "\n".join(f"[{fmt_tc(u['start'])}] Спикер {u['speaker']}: {u['text']}" for u in utts)


def llm(prompt: str) -> str:
    base = os.environ.get("LLM_BASE_URL")
    if base:
        model = os.environ.get("LLM_MODEL", "qwen3.6-plus")
        log(f"ИИ: облако {base} ({model})")
        req = urllib.request.Request(
            base.rstrip("/") + "/chat/completions",
            data=json.dumps({
                "model": model, "temperature": 0.2,
                "messages": [{"role": "user", "content": prompt}],
            }).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {os.environ.get('LLM_API_KEY', '')}"},
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)["choices"][0]["message"]["content"]

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    log(f"ИИ: локально {LOCAL_LLM}")
    model, tok = load(LOCAL_LLM)
    p = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                tokenize=False, enable_thinking=False)
    return generate(model, tok, prompt=p, max_tokens=4000, sampler=make_sampler(temp=0.2))


def parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("ИИ не вернул JSON:\n" + text[:500])
    return json.loads(m.group(0))


def make_protocol(utts: list[dict], date: datetime) -> dict:
    prompt = PROMPT.replace("{date}", f"{date:%d.%m.%Y %H:%M}").replace("{transcript}", transcript_text(utts))
    raw = llm(prompt)
    try:
        return parse_json(raw)
    except (ValueError, json.JSONDecodeError):
        log("ИИ вернул некорректный JSON, пробую ещё раз")
        return parse_json(llm(prompt + "\nВНИМАНИЕ: верни строго валидный JSON."))


# ---------- 5. PDF ----------

def render_pdf(path: Path, date: datetime, duration: float, utts: list[dict], proto: dict | None) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from xml.sax.saxutils import escape

    fonts = Path("/System/Library/Fonts/Supplemental")
    pdfmetrics.registerFont(TTFont("Body", str(fonts / "Arial.ttf")))
    pdfmetrics.registerFont(TTFont("Bold", str(fonts / "Arial Bold.ttf")))

    ink, muted, line, accent = colors.HexColor("#1d1d1f"), colors.HexColor("#6e6e73"), \
        colors.HexColor("#d2d2d7"), colors.HexColor("#f5f5f7")
    body = ParagraphStyle("b", fontName="Body", fontSize=10, leading=14, textColor=ink, alignment=TA_LEFT)
    small = ParagraphStyle("s", parent=body, fontSize=9, leading=12)
    cell_h = ParagraphStyle("ch", parent=small, fontName="Bold")
    meta = ParagraphStyle("m", parent=body, textColor=muted)
    h1 = ParagraphStyle("h1", parent=body, fontName="Bold", fontSize=18, leading=22, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=body, fontName="Bold", fontSize=13, leading=17, spaceBefore=14, spaceAfter=6)
    spk = ParagraphStyle("sp", parent=small, fontName="Bold", textColor=muted, spaceBefore=6)

    def P(text, style=body):
        return Paragraph(escape(str(text if text not in (None, "") else "—")).replace("\n", "<br/>"), style)

    def table(head, rows, widths):
        data = [[P(h, cell_h) for h in head]] + [[P(c, small) for c in r] for r in rows]
        t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), accent),
            ("LINEBELOW", (0, 0), (-1, -1), 0.5, line),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        return t

    p = proto or {}
    n_spk = len({u["speaker"] for u in utts})
    story = [P(p.get("title") or "Запись встречи", h1),
             P(f"{date:%d.%m.%Y, %H:%M} · {fmt_tc(duration)} · спикеров: {n_spk}", meta), Spacer(1, 6)]

    if proto is None:
        story.append(P("Протокол не сформирован (запуск без ИИ). Ниже — полный текст расшифровки.", meta))
    else:
        story += [P("1. Краткое резюме", h2), P(p.get("summary"))]
        story += [P("2. Основные темы обсуждения", h2)]
        topics = p.get("topics") or []
        story.append(table(["№", "Тема", "Краткое содержание", "Таймкоды"],
                           [[i, t.get("title"), t.get("summary"), t.get("timecodes")] for i, t in enumerate(topics, 1)],
                           [10, 45, 90, 25]) if topics else P("—"))
        story += [P("3. Договорённости и решения", h2)]
        ag = p.get("agreements") or []
        story.append(table(["№", "Договорённость", "Краткое содержание"],
                           [[i, a.get("title"), a.get("summary")] for i, a in enumerate(ag, 1)],
                           [10, 60, 100]) if ag else P("—"))
        story += [P("4. Задачи", h2)]
        tasks = p.get("tasks") or []
        story.append(table(["№", "Задача", "Ответственный", "Срок", "Таймкод"],
                           [[i, t.get("task"), t.get("owner"), t.get("deadline"), t.get("timecode")]
                            for i, t in enumerate(tasks, 1)],
                           [10, 75, 35, 30, 20]) if tasks else P("—"))
        story += [P("5. Контекст", h2), P(p.get("context"))]
        story += [P("6. Карточка записи", h2),
                  P(f"Дата записи: {date:%d.%m.%Y %H:%M}"),
                  P(f"Длительность: {fmt_tc(duration)}"),
                  P(f"Количество спикеров: {p.get('speakers_count') or n_spk}"),
                  P(f"Комментарий по качеству: {p.get('quality_comment') or '—'}")]

    story += [P(("7. " if proto else "") + "Полный текст расшифровки", h2)]
    for u in utts:
        story += [P(f"Спикер {u['speaker']} · {fmt_tc(u['start'])}", spk), P(u["text"])]

    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm,
                            title=p.get("title") or "Запись встречи")
    doc.build(story)


# ---------- Сборка ----------

def process(src: Path, speakers: int | None, use_llm: bool, redo: bool) -> Path:
    date = recorded_at(src)
    out = OUT_DIR / f"{date:%Y-%m-%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    tr_file, pr_file = out / "transcript.json", out / "protocol.json"

    if tr_file.exists() and not redo:
        tr = json.loads(tr_file.read_text())
        log(f"{src.name}: расшифровка из кэша")
    else:
        audio = load_audio(src)
        dur = audio.size / SR
        log(f"{src.name}: {fmt_tc(dur)}, расшифровка…")
        t = time.time(); words = transcribe(audio); log(f"  текст готов за {time.time() - t:.0f} с")
        t = time.time(); turns = diarize(audio, speakers); log(f"  спикеры готовы за {time.time() - t:.0f} с")
        tr = {"source": str(src), "duration": dur, "words": words, "turns": turns,
              "utterances": merge(words, turns)}
        tr_file.write_text(json.dumps(tr, ensure_ascii=False, indent=1))
        pr_file.unlink(missing_ok=True)

    proto = None
    if use_llm:
        if pr_file.exists() and not redo:
            proto = json.loads(pr_file.read_text())
            log("  протокол из кэша")
        else:
            t = time.time(); proto = make_protocol(tr["utterances"], date)
            log(f"  протокол готов за {time.time() - t:.0f} с")
            pr_file.write_text(json.dumps(proto, ensure_ascii=False, indent=1))

    pdf = out / f"{out.name}.pdf"
    render_pdf(pdf, date, tr["duration"], tr["utterances"], proto)
    log(f"  PDF: {pdf}")
    return pdf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", type=Path, help=".ops или .wav")
    ap.add_argument("--recorder", action="store_true", help="забрать новые записи с подключённого диктофона")
    ap.add_argument("--speakers", type=int, help="число собеседников, если известно")
    ap.add_argument("--no-llm", action="store_true", help="без протокола, только текст")
    ap.add_argument("--redo", action="store_true", help="пересчитать, игнорируя кэш")
    args = ap.parse_args()
    load_env()

    files = list(args.inputs)
    if args.recorder:
        files += import_from_recorder()
    if not files:
        ap.error("укажите файл или --recorder")
    for f in files:
        process(f, args.speakers, not args.no_llm, args.redo)


if __name__ == "__main__":
    main()
