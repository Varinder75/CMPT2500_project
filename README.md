# MLflow Integration for ML Application

## Project Overview
This project integrates MLflow for tracking machine learning experiments. It contains two main containers:
1. **mlflow-tracking**: A container running the MLflow tracking server.
2. **ml-application**: A container running the ML application and logging data to MLflow.

## Prerequisites
- Docker (v20.10.0 or higher)
- Docker Compose (v1.29.0 or higher)

## Step-by-Step Instructions

### 1. **Build the Docker Images**
To build the Docker images for both containers, use the following command:
```bash
docker-compose build
```
### 2. **Start the Containers**
After building the images, start both containers with:
```bash
docker-compose up
```
This will start:

-> The mlflow-tracking container (exposed on port 5000).

-> The ml-application container, which will log data to MLflow.

### 3. **Access the MLflow UI**
After starting the containers, access the MLflow UI at:
http://localhost:5000
Here, you can track experiments, models, and logs.

### 4. **Log Models to MLflow**
Ensure the ML application logs to the MLflow server by configuring the tracking URI. Example:
``` bash
import mlflow

mlflow.set_tracking_uri("http://mlflow-tracking:5000")
mlflow.start_run()
# Your ML code here...
mlflow.log_metric("accuracy", accuracy)
mlflow.log_param("param", value)
mlflow.end_run()
```
This logs your experiments, metrics, and parameters to the MLflow tracking server.

### 5. **Stop the Containers**
To stop the containers, run:
```bash
docker-compose down
```

### 6. **Troubleshooting**
1. Logs:
If the containers aren't starting as expected, you can check the logs:
``` bash
docker-compose logs
```
This will show the logs for all containers.

2. Common Issues:
-> Port conflict: If port 5000 is in use by another service, you may encounter issues accessing the MLflow UI. You can either stop the service using port 5000 or modify the port mapping in the docker-compose.yml file.

-> Container startup failure: If a container fails to start, check the logs for error messages and ensure all dependencies are installed and correctly configured.

### 7. **Team Members**
Varinder - Project Lead, Docker Configuration, MLflow Integration

Satvir - Backend, ML Application, Docker Setup, Troubleshooting


