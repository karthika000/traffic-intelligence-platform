FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/workspace \
    MPLBACKEND=Agg \
    HF_HOME=/hf_cache

# Heavy dependencies first so changes to shared/ don't reinstall torch.
COPY ml/requirements-runtime.txt ml/requirements-nlp.txt ml/requirements.txt /tmp/ml/
RUN pip install -r /tmp/ml/requirements.txt && rm -rf /tmp/ml

COPY shared/ /tmp/shared/
RUN pip install /tmp/shared && rm -rf /tmp/shared

# ml/ and data/ are bind-mounted here by the ml-trainer compose service.
WORKDIR /workspace
CMD ["python", "-m", "ml.training.train_congestion", "--help"]
