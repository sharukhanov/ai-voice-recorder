# ai-voice-recorder

Своё приложение для диктофона AIREC: записи → текст по спикерам → протокол встречи → PDF.
Цели и план — [docs/PRODUCT.md](docs/PRODUCT.md), разбор устройства — [docs/RESEARCH.md](docs/RESEARCH.md),
шаблон протокола — [docs/SUMMARY_TEMPLATE.md](docs/SUMMARY_TEMPLATE.md).

## Прототип на Mac (Apple Silicon)
```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python tools/meeting.py --recorder              # новые записи с подключённого диктофона → PDF
.venv/bin/python tools/meeting.py captures/ops/FILE.ops   # одна запись
```
Результат: `meetings/<дата_время>/<дата_время>.pdf`.

Модели лежат в `~/ИИ` (ярлык на рабочем столе): `Whisper (расшифровка)`, `Qwen 3.5 9B (протоколы)`,
`Спикеры (деление по голосам)`. Если папки нет — модели Whisper/Qwen скачаются с Hugging Face.

Облачный ИИ вместо локального — файл `.env` в корне (не в git):
```
LLM_BASE_URL=https://.../v1
LLM_MODEL=qwen3.6-plus
LLM_API_KEY=...
```
