### Create input files
    python -m src.create_input_files --csv_file data/raw/UrbanSound8K.csv --data_dir data/signals --input_data_dir data/raw

### Train model
    python -m src.train --train_data data/signals/train --val_data data/signals/val --batch_size 128 --num_epochs 1 --checkpoint checkpoints/checkpoint.pth.tar

### Evaluate model
    python -m src.test --data_dir data/signals/test --run_id 6c6382f62155418ebfcf93d124956ea1 --artifact_path best.pth.tar

### Register model
    python -m src.model_registry --run_id 6c6382f62155418ebfcf93d124956ea1 --artifact_path best.pth.tar --experiment_name SoundClassification

### Serve model
    $env:MLFLOW_TRACKING_URI='http://localhost:5000' # Set the MLflow tracking URI for windows

    mlflow models serve -m models:/model/latest -p 5000 --no-conda 

### Make predictions
    python -m src.prediction --input_file data\raw\fold3\6988-5-0-2.wav