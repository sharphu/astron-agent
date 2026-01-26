FROM python:3.11-slim

WORKDIR /opt/core

ENV PATH=$PATH:/opt/core
ENV PYTHONPATH /opt/core
ENV UV_NO_CACHE=1

RUN pip install uv --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/

COPY core/agent/pyproject.toml ./
COPY core/agent/uv.lock ./

RUN uv sync -i https://mirrors.aliyun.com/pypi/simple/

COPY core/common ./common
COPY core/agent ./agent

CMD ["uv", "run", "agent/main.py"]
