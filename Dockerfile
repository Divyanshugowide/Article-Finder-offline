# Multi-purpose image for Article Finder + Crystal Chat
FROM python:3.11-slim

# System settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python deps first (leverages Docker layer cache)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

# Copy project
COPY . /app

# Default app and port (can be overridden at runtime)
ENV APP_MODULE=app.run_api:app
ENV PORT=8000
EXPOSE 8000

# Optional non-root user
RUN adduser --disabled-password --gecos "" appuser \
 && chown -R appuser:appuser /app
USER appuser

# Start the selected FastAPI app
CMD ["sh", "-c", "uvicorn ${APP_MODULE} --host 0.0.0.0 --port ${PORT}"]

# # ==========================================
# # ✅ FINAL FIXED DOCKERFILE (works on Debian 13 / Trixie)
# # ==========================================

# FROM python:3.10-slim

# WORKDIR /app

# # Use proper ENV syntax
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1

# # ✅ Correct packages for Debian Trixie
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     libgl1 \
#     libglx0 \
#     libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# EXPOSE 8000

# CMD ["uvicorn", "app.run_api:app", "--host", "0.0.0.0", "--port", "8000"]

# ================================================================
# 💎 Crystal AI - Dual Service Dockerfile (API + Chatbot)
# Works on Debian 12/13
# ================================================================

FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------
# 🩵 FIX: Debian 13 APT cleanup bug
# ------------------------------------------------
RUN set -eux; \
    rm -f /etc/apt/apt.conf.d/docker-clean; \
    apt-get clean; \
    apt-get update -o Acquire::CompressionTypes::=gz; \
    apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libgl1 \
        libglx0 \
        poppler-utils \
        supervisor \
        fonts-dejavu-core \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------
# 📦 Install Python dependencies
# ------------------------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------
# 📂 Copy project files
# ------------------------------------------------
COPY . .

# ------------------------------------------------
# 🌐 Expose both services
# ------------------------------------------------
EXPOSE 8000 8001

# ------------------------------------------------
# ⚙️ Supervisor manages both processes
# ------------------------------------------------
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisor.conf"]
