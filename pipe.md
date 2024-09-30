### Create input files
    python src/create_input_files.py --csv_file data/raw/UrbanSound8K.csv --data_dir data/signals --input_data_dir data/raw

### Train model
    python src/train.py --train_data data/signals/train --val_data data/signals/val --batch_size 128 --num_epochs 1 --checkpoint checkpoints/checkpoint.pth.tar

### Evaluate model
    python src/test.py --data_dir data/signals/test --run_id 6c6382f62155418ebfcf93d124956ea1 --artifact_path best.pth.tar

### Register model
    python src/model_registry.py --run_id 6c6382f62155418ebfcf93d124956ea1 --artifact_path best.pth.tar --experiment_name SoundClassification

### Serve model
    mlflow models serve -m models:/model/latest -p 5000 --no-conda 

### Make predictions
    python src/prediction.py --input_file data\raw\fold3\6988-5-0-2.wav