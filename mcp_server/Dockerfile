# Use official Python image
FROM python:3.11-slim

WORKDIR /app

# Create data directory for persistence
RUN mkdir -p /data

COPY main.py ./

RUN pip install fastapi uvicorn pydantic

EXPOSE 10642

# Default command runs HTTP server, but can be overridden for MCP stdio mode
CMD ["python", "main.py"]
