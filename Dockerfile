# ==========================================
# ✅ FINAL FIXED DOCKERFILE (works on Debian 13 / Trixie)
# ==========================================

FROM python:3.10-slim

WORKDIR /app

# Use proper ENV syntax
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ✅ Correct packages for Debian Trixie
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglx0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.run_api:app", "--host", "0.0.0.0", "--port", "8000"]
