FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies required by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY models_dataset1/deployment/requirements.txt /app/models_dataset1/deployment/requirements.txt
RUN pip install --no-cache-dir -r /app/models_dataset1/deployment/requirements.txt

COPY models_dataset1 /app/models_dataset1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "models_dataset1.deployment.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
