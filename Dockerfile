# ═══════════════════════════════════════════════════════════
#  Deepfake Detection System — Hugging Face Spaces Dockerfile
# ═══════════════════════════════════════════════════════════
#
#  Hugging Face Spaces requires port 7860.
#  The backend reads PORT from the environment (defaults to 7860).
#
#  Build context must be the repo root.
# ═══════════════════════════════════════════════════════════

FROM python:3.10-slim

# ── System deps ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────
WORKDIR /app

# ── Copy inference engine (model code + checkpoint) ──────────
COPY ["inference engine/", "./inference engine/"]

# ── Copy backend server ───────────────────────────────────────
COPY backend/ ./backend/

# ── Install Python dependencies ───────────────────────────────
# Install CPU-only PyTorch first (saves ~800 MB vs CUDA build)
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the backend requirements
RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    "uvicorn[standard]>=0.24.0" \
    python-multipart>=0.0.6 \
    timm>=0.9.12 \
    grad-cam>=1.4.8 \
    "opencv-python-headless>=4.8.0" \
    Pillow>=10.0.0 \
    "numpy>=1.24.0" \
    "mediapipe>=0.10.9"

# ── Expose HF Spaces port ─────────────────────────────────────
EXPOSE 7860

# ── Set environment variables ─────────────────────────────────
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# ── Run the FastAPI server ────────────────────────────────────
CMD ["python", "backend/server.py"]
