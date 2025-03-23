# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the necessary files for your application
COPY src/ /app/src/
COPY models/ /app/models/
COPY configs/ /app/configs/

# Expose the port your Flask API will run on
EXPOSE 5000

# Command to run your prediction API
CMD ["python", "src/predict_api.py"]

# Create logs directory in the container
RUN mkdir -p /app/logs

# Set the environment variable for logs
ENV LOG_DIR=/app/logs

# Mount the volume (in your docker-compose or when running the container)
VOLUME /app/logs