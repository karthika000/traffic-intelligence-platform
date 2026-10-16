FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Model runtime first (torch, scikit-learn, MLflow) so code changes don't reinstall it.
COPY ml/requirements-runtime.txt /tmp/requirements-runtime.txt
RUN pip install -r /tmp/requirements-runtime.txt && rm /tmp/requirements-runtime.txt

COPY shared/ /tmp/shared/
RUN pip install /tmp/shared && rm -rf /tmp/shared

COPY services/anomaly_service/requirements.txt .
RUN pip install -r requirements.txt

# The ml package: features, rules and alarm logic shared with training.
COPY ml/ ml/
COPY services/anomaly_service/ .

# The models are tiny: OpenMP/PyTorch defaulting to every vCPU made one congestion prediction
# take 18 s instead of 45 ms. (Set after the installs so changing it keeps the layer cache.)
ENV OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MODEL_THREADS=2

CMD ["python", "main.py"]
