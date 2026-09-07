#!/bin/sh
set -eu

export LOG_FORMAT="${LOG_FORMAT:-json}"

CHROMA_DIR="${CHROMA_DB_DIR:-/app/chroma_db}"
INDEX_DB="$CHROMA_DIR/chroma.sqlite3"
CORPUS_MARKER="$CHROMA_DIR/.corpus.sha256"

CORPUS_HASH="$(python - <<'PY'
from hashlib import sha256

from career_copilot.config import get_settings

settings = get_settings()
digest = sha256()
digest.update(settings.embed_model.encode("utf-8"))
for path in sorted(settings.portfolio_dir.glob("*.md")):
    digest.update(path.name.encode("utf-8"))
    digest.update(path.read_bytes())
print(digest.hexdigest())
PY
)"

NEEDS_REINDEX=0
if [ "${FORCE_REINDEX:-0}" = "1" ]; then
  NEEDS_REINDEX=1
elif [ ! -f "$INDEX_DB" ] || [ ! -f "$CORPUS_MARKER" ]; then
  NEEDS_REINDEX=1
elif [ "$(cat "$CORPUS_MARKER")" != "$CORPUS_HASH" ]; then
  NEEDS_REINDEX=1
fi

if [ "$NEEDS_REINDEX" = "1" ]; then
  echo "Building portfolio index"
  python scripts/build_index.py
  printf '%s' "$CORPUS_HASH" > "$CORPUS_MARKER"
else
  echo "Using current Chroma index"
fi

exec uvicorn career_copilot.api.app:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --proxy-headers \
  --forwarded-allow-ips="${FORWARDED_ALLOW_IPS:-127.0.0.1}"
