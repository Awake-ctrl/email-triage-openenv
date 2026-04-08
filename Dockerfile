# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source ───────────────────────────────────────────────────────────────
COPY env/         ./env/
COPY server.py    .
COPY inference.py .
COPY openenv.yaml .

# ── Hugging Face Spaces runs as non-root user (uid 1000) ─────────────────────
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# ── Expose port 7860 (required by HF Spaces) ─────────────────────────────────
EXPOSE 7860

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Start the environment server ──────────────────────────────────────────────
CMD ["python", "-m", "uvicorn", "server:app", \
     "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
