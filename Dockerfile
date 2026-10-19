FROM python:3.11-slim

# Install system dependencies for headless OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy project specification
COPY pyproject.toml ./

# Install dependencies using uv into system Python
RUN uv pip install --system -r pyproject.toml

# Copy application code
COPY src/ ./src/
COPY main.py ./

ENV PYTHONPATH=/app/src
ENV STORAGE_ROOT=/app/data
ENV DATABASE_PATH=/app/data/letter_worker.db

VOLUME ["/app/data"]

ENTRYPOINT ["python", "-m", "letter_worker.main"]
CMD ["worker"]
