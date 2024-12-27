ARG POETRY_VERSION=1.6.1

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y tzdata
# ENV TZ Asia/Tokyo

RUN apt-get update && \
    apt-get install --yes --no-install-recommends curl g++ libopencv-dev python3.10 python3-pip && \
    rm -rf /var/lib/apt/lists/*
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=${POETRY_VERSION} python3.10 -

ENV APP_HOME /app
WORKDIR $APP_HOME

COPY pyproject.toml poetry.lock ./

ENV PATH="/root/.local/bin:$PATH"
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-root && \
    rm -rf /root/.cache/pypoetry && \
    rm -rf /root/.cache/pip

#use paddlegpu
RUN pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

#for runpod serverless
RUN pip install runpod


COPY . ./
COPY magic-pdf.gpu.json /root/magic-pdf.json

RUN python3.10 download_models.py
#serverless
# CMD ["sh", "-c", "ls && python3.10 serverless.py"]

# Download and set up models
RUN mkdir -p models && \
    # Download PP Detect Model
    curl -L -o models/det_model.tar paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar && \
    tar -xvf models/det_model.tar -C models/ && \
    rm models/det_model.tar && \
    \
    # Download PP Rec Model
    curl -L -o models/rec_model.tar paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar && \
    tar -xvf models/rec_model.tar -C models/ && \
    rm models/rec_model.tar

CMD ["python3.10", "-m", "app.serverless"]

# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000"]
