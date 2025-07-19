# Use official Python image
FROM python:3.11-slim

WORKDIR /app

# Create data directory for persistence
RUN mkdir -p /data

COPY main.py ./

RUN pip install fastapi uvicorn pydantic

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
