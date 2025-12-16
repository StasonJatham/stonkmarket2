FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps for timezone data and certificates
RUN apt-get update \ 
    && apt-get install -y --no-install-recommends ca-certificates tzdata build-essential \ 
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DB_PATH=/data/dips.sqlite
RUN mkdir -p /data

EXPOSE 8000
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "main:app", "--bind", "0.0.0.0:8000", "--access-logfile", "-", "--forwarded-allow-ips", "*"]
