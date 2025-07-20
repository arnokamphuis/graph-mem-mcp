# Use official Python image
FROM python:3.11-slim

# Create non-root user
RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser appuser

WORKDIR /app

# Create data directory for persistence and set ownership
RUN mkdir -p /data && \
    chown -R appuser:appuser /app /data

# Copy requirements first for better layer caching
COPY requirements.txt ./

# Update pip and install dependencies as root (suppress root user warnings for container build)
RUN pip install --upgrade pip --disable-pip-version-check --no-warn-script-location --root-user-action=ignore && \
    pip install --no-cache-dir -r requirements.txt --disable-pip-version-check --no-warn-script-location --root-user-action=ignore

# Copy application files and set ownership
COPY main.py ./
COPY templates/ ./templates/
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

EXPOSE 10642

# Default command runs HTTP server, but can be overridden for MCP stdio mode
CMD ["python", "main.py"]
