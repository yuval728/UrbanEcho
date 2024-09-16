import torch
import mlflow
import argparse
import utils as ut
import warnings
warnings.filterwarnings("ignore")

    
def get_model_from_artifact_checkpoint(run_id: str, artifact_path: str):
    mlflow_client = mlflow.tracking.MlflowClient()
    model_path = mlflow_client.download_artifacts(run_id, artifact_path)
    model, _, _, _, _ = ut.load_checkpoint(model_path)
    return model

def create_model_signature(model, input_shape, artifact_path='model'):
    input_sample = torch.rand(1, *input_shape)
    signature = mlflow.models.infer_signature(input_sample.detach().numpy(), model(input_sample).detach().numpy())

    mlflow.pytorch.log_model(model, artifact_path, signature=signature)
    
def register_model(run_id: str, artifact_path: str, input_shape: tuple, model_name: str='model', tracking_uri='http://localhost:5000', experiment_name='Default'):
    mlflow.set_experiment(experiment_name)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.start_run(run_id=run_id, nested=True)
    
    model = get_model_from_artifact_checkpoint(run_id, artifact_path)
    create_model_signature(model, input_shape)
    
    mlflow.register_model(f"runs:/{run_id}/{model_name}", model_name)
    
def parse_args():
    parser = argparse.ArgumentParser(description='Register a model')
    parser.add_argument('--run_id', type=str, required=True, help='Run ID of the model')
    parser.add_argument('--artifact_path', type=str, required=True, help='Artifact path of the model')
    parser.add_argument('--input_shape', type=int, default=(1,50), help='Input shape of the model', nargs='+')
    parser.add_argument('--model_name', type=str, default='model', help='Name of the model')
    parser.add_argument('--tracking_uri', type=str, default='http://localhost:5000', help='URI of the tracking server')
    parser.add_argument('--experiment_name', type=str, default='Default', help='Name of the experiment')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    register_model(args.run_id, args.artifact_path, args.input_shape, args.model_name, args.tracking_uri, args.experiment_name)
    
    
    
