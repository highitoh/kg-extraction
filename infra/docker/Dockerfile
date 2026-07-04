FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application code
COPY app/ /app/

# Install uv (fast Python package manager)
RUN pip install uv

# Install Python dependencies
RUN cd /app && uv sync

# Set Python path
ENV PYTHONPATH=/app

# Set default storage backend to S3 for AWS
ENV STORAGE_BACKEND=s3

# Entry point is batch_runner.py
ENTRYPOINT ["uv", "run", "python", "src/batch_runner.py"]
