#!/usr/bin/env python3
"""Конвертер записей диктофона AIREC (.ops) в WAV.

Формат .ops (разобран 2026-11-21, см. docs/RESEARCH.md):
  - файл — подряд идущие слоты по 80 байт, общего заголовка нет;
  - каждый слот — один пакет Opus (SILK wideband, 16 кГц, моно, 20 мс):
      0x4b 0x41 <pad> ...  — TOC code 3, 1 кадр, <pad> байт нулевой добивки в конце;
      0x4b 0x01 ...        — TOC code 3, 1 кадр, без добивки;
      0x48 ...             — TOC code 0, один кадр на весь слот;
  - 80 байт × 50 пакетов/с = 32 кбит/с (CBR);
  - последний слот в файле может быть обрезан — его пропускаем.

Использование:
    python3 tools/ops2wav.py FILE.ops [...]          # WAV рядом с исходником
    python3 tools/ops2wav.py DIR                     # все .ops в папке
    python3 tools/ops2wav.py -o OUTDIR FILE_OR_DIR   # WAV в отдельную папку
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import av

SLOT = 80
OUT_RATE = 16000


def decode_ops(data: bytes) -> bytes:
    """Возвращает PCM s16le, моно, 16 кГц."""
    dec = av.CodecContext.create("libopus", "r")
    res = av.AudioResampler(format="s16", layout="mono", rate=OUT_RATE)
    pcm = bytearray()
    bad = 0
    for i in range(0, len(data) - SLOT + 1, SLOT):
        try:
            frames = dec.decode(av.Packet(data[i:i + SLOT]))
        except av.FFmpegError:
            bad += 1
            continue
        for fr in frames:
            for out in res.resample(fr):
                pcm += out.to_ndarray().tobytes()
    for out in res.resample(None):
        pcm += out.to_ndarray().tobytes()
    if bad:
        print(f"  предупреждение: {bad} битых пакетов пропущено", file=sys.stderr)
    return bytes(pcm)


def convert(src: Path, out_dir: Path | None) -> Path:
    dst = (out_dir or src.parent) / (src.stem + ".wav")
    pcm = decode_ops(src.read_bytes())
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(OUT_RATE)
        w.writeframes(pcm)
    print(f"{src.name} -> {dst}  ({len(pcm) / 2 / OUT_RATE:.1f} с)")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, help="папка для WAV")
    args = ap.parse_args()
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    files = []
    for p in args.inputs:
        files += sorted(p.glob("*.ops")) if p.is_dir() else [p]
    if not files:
        sys.exit("нет .ops файлов")
    for f in files:
        convert(f, args.out)


if __name__ == "__main__":
    main()
