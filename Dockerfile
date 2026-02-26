FROM python:3.10-slim

WORKDIR /app

ENV TRANSFORMERS_CACHE=/app/cache

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    gcc \
    curl \
    pkg-config \
    libsentencepiece-dev \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

EXPOSE 7860

CMD ["python", "src/app.py"]