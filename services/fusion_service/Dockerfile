FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Build context is the repo root (see docker-compose.yml).
COPY shared/ /tmp/shared/
RUN pip install /tmp/shared && rm -rf /tmp/shared

COPY services/fusion_service/requirements.txt .
RUN pip install -r requirements.txt

COPY services/fusion_service/ .

CMD ["python", "main.py"]
