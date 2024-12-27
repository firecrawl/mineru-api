ARG POETRY_VERSION=1.6.1

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
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
RUN pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/

COPY . ./
COPY magic-pdf.gpu.json /root/magic-pdf.json

RUN python3.10 download_models.py


CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000"]
