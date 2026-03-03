ARG POETRY_VERSION=1.6.1

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04
# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y tzdata
# ENV TZ Asia/Tokyo

RUN apt-get update && \
    apt-get install --yes --no-install-recommends patch curl g++ libopencv-dev python3 python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*


RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=${POETRY_VERSION} python3 -

ENV APP_HOME /app
WORKDIR $APP_HOME

COPY pyproject.toml poetry.lock ./

ENV PATH="/root/.local/bin:$PATH"
RUN poetry config virtualenvs.in-project true && \
    poetry lock --no-interaction && \
    poetry install --no-interaction --no-root && \
    rm -rf /root/.cache/pypoetry && \
    rm -rf /root/.cache/pip

# Patch mineru to support batch (batch_ratio=32 for 24GB VRAM, force OCR-det batching)
COPY patch/mineru_batch.patch /tmp/mineru_batch.patch
RUN patch .venv/lib/python3.*/site-packages/mineru/backend/pipeline/pipeline_analyze.py < /tmp/mineru_batch.patch
# Cap OCR-det forward batch size to N=1 so CUDA kernels match warmup cache
COPY patch/batch_analyze_det_bs.patch /tmp/batch_analyze_det_bs.patch
RUN patch .venv/lib/python3.*/site-packages/mineru/backend/pipeline/batch_analyze.py < /tmp/batch_analyze_det_bs.patch
# Increase Layout/MFD batch sizes from 1 to 8 for better GPU utilization
COPY patch/batch_sizes.patch /tmp/batch_sizes.patch
RUN patch .venv/lib/python3.*/site-packages/mineru/backend/pipeline/batch_analyze.py < /tmp/batch_sizes.patch
# Enable CUDA for wired table UNet model (was CPU-only)
COPY patch/wired_table_cuda.patch /tmp/wired_table_cuda.patch
RUN patch .venv/lib/python3.*/site-packages/mineru/model/table/rec/unet_table/utils.py < /tmp/wired_table_cuda.patch
# Add the virtual environment's bin directory to PATH
ENV PATH="$APP_HOME/.venv/bin:$PATH"

#use paddlegpu
# RUN pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

#for runpod serverless
# RUN pip install runpod


COPY . ./

RUN /bin/bash -c "mineru-models-download -s huggingface -m pipeline"

# Healthcheck: RunPod serverless starts on port 8000 after warmup completes.
# start-period covers model loading + CUDA kernel warmup (~10 min).
HEALTHCHECK --interval=10s --timeout=5s --start-period=600s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set the entry point to activate the virtual environment and run the command line tool
ENTRYPOINT ["/bin/bash", "-c", "export MINERU_MODEL_SOURCE=local && python3 -m app.serverless"]
# CMD ["python3", "-m", "app.serverless"]

# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000"]
