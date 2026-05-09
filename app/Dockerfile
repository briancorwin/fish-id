FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8080

# Single worker keeps in-process rate limiter state consistent across requests
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "120", "main:app"]
