# Multi-stage build -- the React frontend is compiled to static files in
# stage 1, then copied into the Python image in stage 2. The final image needs no
# Node.js at all, and the FastAPI app (src/career_copilot/api/app.py) serves both the
# API and the built frontend on the same origin, so there's nothing else to deploy.

# ---- Stage 1: build the React frontend ----
FROM node:22-slim AS frontend-build
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Python backend, serving the built frontend ----
FROM python:3.11-slim AS backend
WORKDIR /app

# chromadb pulls in dependencies (onnxruntime, hnswlib, ...) that occasionally need a
# compiler if no prebuilt wheel matches this exact platform/Python combo -- keeping
# build-essential around is cheap insurance against a pip install that would otherwise
# fail deep inside a transitive dependency with no obvious fix.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

COPY scripts/ ./scripts/
COPY docker/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

# Built frontend from stage 1. api/app.py mounts this directory as static files (via
# FRONTEND_DIST_DIR) once it exists -- unset in local dev, where Vite's own dev server
# serves the frontend instead (see frontend/vite.config.js's proxy).
COPY --from=frontend-build /build/frontend/dist ./frontend_dist
ENV FRONTEND_DIST_DIR=/app/frontend_dist

# Where the Chroma index and SQLite history persist inside the container -- see
# docker/entrypoint.sh for why the index gets rebuilt on boot.
ENV CHROMA_DB_DIR=/app/chroma_db
ENV HISTORY_DB_PATH=/app/history.db

EXPOSE 8000
ENTRYPOINT ["./entrypoint.sh"]
