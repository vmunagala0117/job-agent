# ------------------------------------------------------------------
# Job Agent — Production Container Image
# Multi-stage build for Azure Web App / Container App deployment
# ------------------------------------------------------------------

FROM python:3.13-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (PostgreSQL client libs for psycopg2)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libpq-dev gcc && \
    rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------
# Dependencies layer (cached unless requirements change)
# ------------------------------------------------------------------
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------
# Application layer
# ------------------------------------------------------------------
FROM deps AS app

COPY pyproject.toml .
COPY src/ src/

# Install the project itself (editable not needed in production)
RUN pip install --no-cache-dir -e .

# Expose port (Azure Web App expects 8080 by default)
EXPOSE 8080

# Health check for Container App readiness probes
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Run with uvicorn — single worker, let Azure handle scaling
CMD ["uvicorn", "job_agent.webapp:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]
