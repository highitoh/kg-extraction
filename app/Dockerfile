FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies using uv sync (including dev dependencies for devcontainer)
RUN uv sync --frozen --no-install-project

# Workspace
RUN mkdir -p /workspace
WORKDIR /workspace