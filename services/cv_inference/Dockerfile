FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Ultralytics settings and downloads live in the weights volume, not the image.
    YOLO_CONFIG_DIR=/weights/ultralytics \
    # Never pip-install packages at runtime (Ultralytics does that for missing extras).
    YOLO_AUTOINSTALL=false

WORKDIR /app

# Heavy dependencies first, so changes to shared/ don't re-download ~1 GB of wheels.
# Build context is the repo root (see docker-compose.yml).
COPY services/cv_inference/requirements.txt .
# Ultralytics depends on the GUI build of OpenCV, which needs libGL; swap in the headless build.
RUN pip install -r requirements.txt \
    && pip uninstall -y opencv-python \
    && pip install --no-deps --force-reinstall "opencv-python-headless>=4.10,<5" \
    && python -c "import cv2, torch, ultralytics, openvino; print('cv2', cv2.__version__, 'torch', torch.__version__, 'ultralytics', ultralytics.__version__, 'openvino', openvino.__version__)"

COPY shared/ /tmp/shared/
RUN pip install /tmp/shared && rm -rf /tmp/shared

COPY services/cv_inference/ .

CMD ["python", "main.py"]
