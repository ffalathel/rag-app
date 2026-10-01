FROM python:3.11-slim

# opencv (pulled in by docling's OCR) needs these shared libs, absent from -slim
RUN apt-get update && apt-get install -y --no-install-recommends libxcb1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    GRADIO_ANALYTICS_ENABLED=False \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# CPU-only torch first, so sentence-transformers doesn't pull the CUDA build (~2 GB)
RUN pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user pyproject.toml ./
COPY --chown=user src ./src
RUN pip install --no-cache-dir --user .

# Fetch every model at build time so a wake-up loads from disk. Converting the
# fixture pulls exactly the Docling models (layout, tables, OCR) runtime uses.
COPY --chown=user tests/fixtures/cfpb_closing_disclosure.pdf /tmp/warm.pdf
RUN python -c "from docuchat.config import Settings; \
from docuchat.models import get_embed_model, get_cross_encoder; \
from docuchat.ingest import load_pdf; \
cfg = Settings(); get_embed_model(cfg); get_cross_encoder(cfg); load_pdf('/tmp/warm.pdf', cfg)"

COPY --chown=user eval/corpus ./eval/corpus
COPY --chown=user eval/nodes ./eval/nodes
COPY --chown=user eval/questions.yaml ./eval/questions.yaml

EXPOSE 7860
CMD ["uvicorn", "docuchat.api:app", "--host", "0.0.0.0", "--port", "7860", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
