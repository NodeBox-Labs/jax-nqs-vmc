FROM python:3.11-slim

# Install system dependencies needed by scipy/jax
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the package in editable mode
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
RUN pip install --no-cache-dir -e ".[dev]"

# Copy the rest of the project
COPY . .

# Default command runs the test suite
CMD ["pytest", "-q", "-m", "not slow"]