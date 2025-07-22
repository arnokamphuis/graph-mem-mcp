# Use official Python image
FROM python:3.11-slim

# Create non-root user with home directory
RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser -m -d /home/appuser -s /bin/bash appuser

WORKDIR /app

# Create data directory for persistence and set ownership
RUN mkdir -p /data && \
    mkdir -p /app/.cache/huggingface && \
    chmod -R 755 /app && \
    chown -R appuser:appuser /app /data

# Copy requirements first for better layer caching
COPY requirements.txt ./

# Update pip and install dependencies as root (suppress root user warnings for container build)
RUN pip install --upgrade pip --disable-pip-version-check --no-warn-script-location --root-user-action=ignore && \
    pip install -r requirements.txt --disable-pip-version-check --no-warn-script-location --root-user-action=ignore && \
    python -m spacy download en_core_web_sm

# Copy application files and Phase 1-3 components
COPY main.py ./
COPY knowledge_graph_processor.py ./
COPY enhanced_knowledge_processor.py ./
COPY templates/ ./templates/
COPY core/ ./core/
COPY storage/ ./storage/
COPY extraction/ ./extraction/
COPY quality/ ./quality/
COPY migration/ ./migration/
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set environment variables for model caching (using HF_HOME instead of deprecated TRANSFORMERS_CACHE)
ENV HF_HOME=/app/.cache/huggingface
ENV HF_DATASETS_CACHE=/app/.cache/huggingface/datasets

EXPOSE 10642

# Default command runs HTTP server, but can be overridden for MCP stdio mode
CMD ["python", "main.py"]
