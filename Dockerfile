# Multi-stage build for lurus-memorus REST API service
# Stage 1: build wheel
FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml README-zh.md LICENSE NOTICE ./
COPY memorus/ ./memorus/

RUN pip install --no-cache-dir build && \
    python -m build --wheel --outdir /dist

# Stage 2: runtime
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install runtime extras: api + onnx (offline embeddings) + team federation
# Note: glob expansion before extras requires shell substitution
COPY --from=builder /dist/*.whl /tmp/
RUN WHL=$(ls /tmp/*.whl | head -1) && \
    pip install --no-cache-dir "${WHL}[api,onnx,team,llm]" && \
    rm -f /tmp/*.whl

# Data directory for SQLite and ONNX model cache
# UID 65534 = nobody in Python slim images; create /data owned by that user
RUN mkdir -p /data && chown 65534:65534 /data

ENV MEMORUS_DATA_DIR=/data

VOLUME ["/data"]

USER 65534

EXPOSE 8880

CMD ["memorus-api", "--host", "0.0.0.0", "--port", "8880"]
