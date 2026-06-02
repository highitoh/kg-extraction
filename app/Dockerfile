FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim

# Install dependencies using uv
COPY requirements.txt /tmp/pip-tmp/
RUN uv pip install --system -r /tmp/pip-tmp/requirements.txt \
   && rm -rf /tmp/pip-tmp

# Workspace
RUN mkdir -p /workspace