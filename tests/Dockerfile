# Test runner image. The repo is bind-mounted at /repo and PYTHONPATH points at the live
# shared/src, so code changes need no rebuild; dependency changes do (test.ps1 rebuilds).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/repo/shared/src \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    YOLO_AUTOINSTALL=false

# Service dependencies first (heavy: torch, OpenVINO), so changes to shared/ rebuild only
# the small layer below. Same OpenCV swap as the cv_inference image: Ultralytics pulls in
# the GUI build, which needs libGL.
COPY services/video_ingest/requirements.txt /tmp/requirements/video_ingest.txt
COPY services/event_simulator/requirements.txt /tmp/requirements/event_simulator.txt
COPY services/cv_inference/requirements.txt /tmp/requirements/cv_inference.txt
COPY ml/requirements-runtime.txt /tmp/requirements/ml-runtime.txt
RUN pip install -r /tmp/requirements/video_ingest.txt -r /tmp/requirements/event_simulator.txt \
        -r /tmp/requirements/cv_inference.txt -r /tmp/requirements/ml-runtime.txt \
    && pip uninstall -y opencv-python \
    && pip install --no-deps --force-reinstall "opencv-python-headless>=4.10,<5" \
    && rm -rf /tmp/requirements

COPY shared/ /tmp/shared/
RUN pip install "/tmp/shared[dev]" && rm -rf /tmp/shared

WORKDIR /repo
ENTRYPOINT ["python", "-m", "pytest", "-p", "no:cacheprovider"]
