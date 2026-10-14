# Test runner image. The repo is bind-mounted at /repo and PYTHONPATH points at the live
# shared/src, so code changes need no rebuild; dependency changes do (test.ps1 rebuilds).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/repo/shared/src

COPY shared/ /tmp/shared/
COPY services/video_ingest/requirements.txt /tmp/requirements/video_ingest.txt
RUN pip install "/tmp/shared[dev]" -r /tmp/requirements/video_ingest.txt && rm -rf /tmp/shared /tmp/requirements

WORKDIR /repo
ENTRYPOINT ["python", "-m", "pytest", "-p", "no:cacheprovider"]
