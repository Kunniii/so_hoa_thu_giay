FROM python:3.11-slim

# Install minimal OS dependencies for OpenCV headless and image decoders
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged worker user
RUN useradd -m -u 1000 -s /bin/bash worker

WORKDIR /app

# Install uv for ultra-fast dependency management
RUN pip install --no-cache-dir uv

# Copy project specification and readme
COPY pyproject.toml README.md ./

# Copy source code and entrypoint
COPY src/ ./src/
COPY main.py ./

# Install project as an optimized package
RUN uv pip install --system --no-cache .

# Set up storage directory and permissions
RUN mkdir -p /app/data && chown -R worker:worker /app

USER worker

ENV PYTHONPATH=/app/src
ENV STORAGE_ROOT=/app/data
ENV DATABASE_PATH=/app/data/letter_worker.db
ENV PYTHONUNBUFFERED=1

VOLUME ["/app/data"]

ENTRYPOINT ["letter-worker"]
CMD ["worker", "--db", "/app/data/letter_worker.db", "--storage", "/app/data"]
