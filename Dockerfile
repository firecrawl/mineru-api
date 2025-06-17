ARG POETRY_VERSION=1.6.1

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04
# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y tzdata
# ENV TZ Asia/Tokyo

RUN apt-get update && \
    apt-get install --yes --no-install-recommends curl g++ libopencv-dev python3 python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*


RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=${POETRY_VERSION} python3 -

ENV APP_HOME /app
WORKDIR $APP_HOME

COPY pyproject.toml poetry.lock ./

ENV PATH="/root/.local/bin:$PATH"
RUN poetry config virtualenvs.in-project true && \
    poetry install --no-interaction --no-root && \
    rm -rf /root/.cache/pypoetry && \
    rm -rf /root/.cache/pip

# Add the virtual environment's bin directory to PATH
ENV PATH="$APP_HOME/.venv/bin:$PATH"

#use paddlegpu
# RUN pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

#for runpod serverless
# RUN pip install runpod


COPY . ./

RUN /bin/bash -c "mineru-models-download -s huggingface -m pipeline"

# Set the entry point to activate the virtual environment and run the command line tool
ENTRYPOINT ["/bin/bash", "-c", "export MINERU_MODEL_SOURCE=local && python3 -m app.serverless"]
# CMD ["python3", "-m", "app.serverless"]

# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000"]
