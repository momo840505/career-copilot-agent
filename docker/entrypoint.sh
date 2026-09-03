#!/bin/sh
# Phase 7c: container entrypoint.
#
# Deployment platforms (Render included) typically give the container an EPHEMERAL
# filesystem -- a fresh, empty disk on every deploy/restart -- so the Chroma index
# built by `scripts/build_index.py` can't be assumed to already be there. This script
# rebuilds it on boot before starting the API.
#
# The rebuild is skipped if a Chroma index is already found on disk (e.g. a local
# `docker run` with a mounted volume, or a platform with a persistent disk add-on) --
# that avoids paying for OpenAI embedding calls on every restart when nothing has
# changed. Set FORCE_REINDEX=1 to rebuild anyway (e.g. after editing the portfolio
# markdown files under src/career_copilot/data/portfolio/).
set -e

# Phase 7e: structured (one-JSON-object-per-line) logs in the container by default --
# see career_copilot/observability.py. Render's Logs tab just tails stdout, so this is
# what makes those lines greppable/parseable instead of free-text. Local dev (running
# scripts/run_api.py directly, never through this script) keeps the human-readable
# default. Override with -e LOG_FORMAT=plain on `docker run` if you want to read
# container logs by eye instead.
export LOG_FORMAT="${LOG_FORMAT:-json}"

CHROMA_DIR="${CHROMA_DB_DIR:-/app/chroma_db}"
INDEX_MARKER="$CHROMA_DIR/chroma.sqlite3"

if [ -f "$INDEX_MARKER" ] && [ "$FORCE_REINDEX" != "1" ]; then
  echo "==> Chroma index already present at $CHROMA_DIR, skipping rebuild (set FORCE_REINDEX=1 to force)."
else
  echo "==> Building RAG index (career_copilot portfolio docs -> Chroma)..."
  python scripts/build_index.py
fi

echo "==> Starting API server on port ${PORT:-8000}..."
exec uvicorn career_copilot.api.app:app --host 0.0.0.0 --port "${PORT:-8000}"
