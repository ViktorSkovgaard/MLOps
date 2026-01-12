# FROM ghcr.io/astral-sh/uv:python3.11.7-alpine AS base

# RUN uv sync --frozen --no-install-project

# RUN uv sync --frozen


# Base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# ENV UV_LINK_MODE=copy
# RUN --mount=type=cache,target=/root/.cache/uv uv sync

COPY uv.lock uv.lock

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY reports/figures/ reports/figures/

WORKDIR /
RUN uv sync --locked --no-cache

ENTRYPOINT ["uv", "run", "src/my_project/train.py"]
