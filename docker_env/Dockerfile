# Dockerfile for Universal Training Scripts
# Supports execution of any script in universal_training_script folder
# Maximum execution time: 24 hours

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Note: pennylane 0.31.1 requires numpy<1.24, so we use numpy 1.23.5
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    pandas==2.0.3 \
    scikit-learn==1.3.0 \
    torch==2.0.1 \
    torchvision==0.15.2 \
    pennylane==0.31.1 \
    matplotlib==3.7.2 \
    pillow==10.0.0 \
    xgboost==2.0.0

# Copy universal_training_script folder
COPY universal_training_script /app/universal_training_script

# Create directory for input/output files
RUN mkdir -p /app/data /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default command (can be overridden)
CMD ["python", "--version"]

