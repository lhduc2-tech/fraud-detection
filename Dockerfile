FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for LightGBM and PostgreSQL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5002

ENV PORT=5002
ENV PYTHONUNBUFFERED=1

CMD ["python", "app_server.py"]
