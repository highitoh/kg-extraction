FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim

WORKDIR /app

# Copy dependency files for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies using uv sync (including dev dependencies for devcontainer)
RUN uv sync --frozen --no-install-project

# Workspace
RUN mkdir -p /workspace
WORKDIR /workspace