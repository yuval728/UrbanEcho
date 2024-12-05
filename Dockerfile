# Use the official MLflow image as a base
FROM ghcr.io/mlflow/mlflow:latest

# Set environment variables for MLflow tracking URI
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000

# Install the required dependencies
COPY src src
COPY model_requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that MLflow will use to serve the model
EXPOSE 5000

# Command to serve the registered model
CMD ["mlflow", "models", "serve", "-m", "models:/UrbanEchoModel/latest", "-p", "5000", "--no-conda"]