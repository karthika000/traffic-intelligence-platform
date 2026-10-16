FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/hf_cache

WORKDIR /app

# Build context is the repo root (see docker-compose.yml). Model runtime first (torch,
# transformers, MLflow) so code changes don't reinstall it.
COPY ml/requirements-runtime.txt ml/requirements-nlp.txt /tmp/ml/
RUN pip install -r /tmp/ml/requirements-runtime.txt -r /tmp/ml/requirements-nlp.txt \
    && rm -rf /tmp/ml

COPY shared/ /tmp/shared/
RUN pip install /tmp/shared && rm -rf /tmp/shared

COPY services/summarizer/requirements.txt .
RUN pip install -r requirements.txt

# The ml package: fact context, prompts, template, faithfulness guard and the pyfunc wrapper.
COPY ml/ ml/
COPY services/summarizer/ .

# Same reason as the other model services: PyTorch on every vCPU is slower for one request.
ENV OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MODEL_THREADS=2 TOKENIZERS_PARALLELISM=false

CMD ["python", "main.py"]
