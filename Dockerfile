FROM node:22-slim AS frontend-build
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim AS python-build
WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir .

FROM python:3.11-slim AS runtime
WORKDIR /app

COPY --from=python-build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY scripts/ ./scripts/
COPY docker/entrypoint.sh ./entrypoint.sh
COPY --from=frontend-build /build/frontend/dist ./frontend_dist

ENV FRONTEND_DIST_DIR=/app/frontend_dist
ENV CHROMA_DB_DIR=/app/chroma_db
ENV HISTORY_DB_PATH=/app/history.db

RUN chmod +x ./entrypoint.sh \
    && mkdir -p /app/chroma_db \
    && useradd --create-home --shell /bin/false appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000
ENTRYPOINT ["./entrypoint.sh"]
