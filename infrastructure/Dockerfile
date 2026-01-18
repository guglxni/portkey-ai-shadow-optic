# Shadow-Optic Dockerfile
# Multi-stage build for production-ready container

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install uv

# Copy dependency files
COPY requirements.txt pyproject.toml ./

# Install dependencies
RUN uv pip install --system --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 2: Production
# =============================================================================
FROM python:3.11-slim as production

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/

# Set Python path
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Create non-root user
RUN useradd -m -u 1000 shadow-optic
USER shadow-optic

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import shadow_optic; print('healthy')" || exit 1

# Default command (can be overridden)
CMD ["python", "-m", "shadow_optic.worker"]

# =============================================================================
# Stage 3: Development
# =============================================================================
FROM production as development

USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    mypy \
    ruff \
    ipython

# Install Temporal CLI for development
RUN curl -sSf https://temporal.download/cli.sh | sh

USER shadow-optic

CMD ["python", "-m", "shadow_optic.worker"]
