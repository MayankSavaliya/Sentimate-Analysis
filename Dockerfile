FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p data models/saved_model

# Set environment variables for memory optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV CUDA_VISIBLE_DEVICES=-1

# Expose port
EXPOSE $PORT

# Command to run the application
CMD gunicorn --workers=1 --threads=2 --timeout=0 --bind 0.0.0.0:$PORT app:app
