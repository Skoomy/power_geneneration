# Use Python 3.9 as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY data/ data/
COPY docs/ docs/
COPY *.md ./
COPY *.png ./

# Create directories for outputs
RUN mkdir -p /app/outputs /app/models /app/logs

# Set permissions
RUN chmod -R 755 /app

# Expose ports for Jupyter and potential web services
EXPOSE 8888 8080 5000

# Default command - can be overridden in docker-compose
CMD ["python", "-c", "import src; print('Power Generation Framework ready!')"]