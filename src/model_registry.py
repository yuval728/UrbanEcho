import torch
import mlflow
import argparse
import utils as ut
import librosa
import numpy as np
import base64
from io import BytesIO
from pydub import AudioSegment
import warnings
warnings.filterwarnings("ignore")

class UrbanEcho(mlflow.pyfunc.PythonModel):
    def __init__(self, n_mfcc, classes):
        self.n_mfcc = n_mfcc
        self.classes = classes
    
    def load_context(self, context):
        self.model = torch.jit.load(context.artifacts['model']) 
        self.model.eval()
        # self.n_mfcc = context.artifacts['n_mfcc']
        # self.classes = context.artifacts['classes']
        
    def preprocess(self, input):
    # Decode base64 audio data
        audio_data = base64.b64decode(input)
        audio = AudioSegment.from_file(BytesIO(audio_data), format="wav")
        
        samples = np.array(audio.get_array_of_samples())
    
    # If stereo, take only one channel
        if audio.channels > 1:
            samples = samples.reshape((-1, audio.channels))[:, 0]
        
        # Normalize to range [-1.0, 1.0]
        samples = samples / np.iinfo(samples.dtype).max
        
        # Use librosa to process the numpy array
        scaled_feature = librosa.feature.mfcc(y=samples, sr=audio.frame_rate, n_mfcc=self.n_mfcc)
        # Convert to the appropriate feature
        scaled_feature = np.mean(scaled_feature.T,axis=0)
        scaled_feature = torch.tensor(scaled_feature).float()
        # print(scaled_feature.shape)
        scaled_feature = scaled_feature.unsqueeze(0)
        return scaled_feature
    
    def postprocess(self, input):
        prediction = torch.nn.functional.softmax(input, dim=1)
        # print(prediction)
        return {'class': self.classes[prediction.argmax().item()], 'confidence': prediction.max().item(), 'probabilities': prediction.tolist()}
    
    def predict(self, context, model_input):
        with torch.inference_mode():
            preprocess_input = self.preprocess(model_input)
            output = self.model(preprocess_input.unsqueeze(1))
            return self.postprocess(output)
    
    
def get_model_from_artifact_checkpoint(run_id: str, artifact_path: str):
    mlflow_client = mlflow.tracking.MlflowClient()
    model_path = mlflow_client.download_artifacts(run_id, artifact_path)
    model, _, _, _, _ = ut.load_checkpoint(model_path)
    return model

def create_model_signature(model, input_shape):
    input_sample = torch.rand(1, *input_shape)
    signature = mlflow.models.infer_signature(input_sample.detach().numpy(), model(input_sample).detach().numpy())
    return signature
    
def convert_model_to_jit(model, input_shape, model_name='model'):
    input_sample = torch.rand(1, *input_shape)
    traced_model = torch.jit.trace(model, input_sample)
    torch.jit.save(traced_model, f"{model_name}.pt")  # Save the traced model
    return f"{model_name}.pt"  # Return the path to the saved traced model
    
def register_model(run_id: str, artifact_path: str, input_shape: tuple, classes: list, model_name: str='model', tracking_uri='http://localhost:5000', experiment_name='Default'):
    mlflow.set_experiment(experiment_name)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.start_run(run_id=run_id, nested=True)
    
    # Load model and prepare signature
    model = get_model_from_artifact_checkpoint(run_id, artifact_path)
    # signature = create_model_signature(model, input_shape)
    
    # Convert model to TorchScript and save the traced model
    traced_model_path = convert_model_to_jit(model, input_shape, model_name)
    
    # Log the model with mlflow.pyfunc
    mlflow.pyfunc.log_model(
        artifact_path=model_name, 
        python_model=UrbanEcho(n_mfcc=input_shape[1], classes=classes), 
        artifacts={'model': traced_model_path}, 
        # signature=signature
    )
    
    # Register the model in the model registry
    mlflow.register_model(f"runs:/{run_id}/{model_name}", model_name)
    
def parse_args():
    parser = argparse.ArgumentParser(description='Register a model')
    parser.add_argument('--run_id', type=str, required=True, help='Run ID of the model')
    parser.add_argument('--artifact_path', type=str, required=True, help='Artifact path of the model')
    parser.add_argument('--input_shape', type=int, default=[1, 50], help='Input shape of the model', nargs='+')
    parser.add_argument('--classes', type=str, default='classes.txt', help='Path to the classes file')
    parser.add_argument('--model_name', type=str, default='model', help='Name of the model')
    parser.add_argument('--tracking_uri', type=str, default='http://localhost:5000', help='URI of the tracking server')
    parser.add_argument('--experiment_name', type=str, default='Default', help='Name of the experiment')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    with open(args.classes, 'r') as f:
        classes = f.readlines()
        classes = [c.strip() for c in classes]
    
    register_model(run_id=args.run_id, artifact_path=args.artifact_path, input_shape=tuple(args.input_shape), model_name=args.model_name, tracking_uri=args.tracking_uri, experiment_name=args.experiment_name, classes=classes)
    
    print(f"Model {args.model_name} registered successfully! Run ID: {args.run_id} Experiment: {args.experiment_name} Tracking URI: {args.tracking_uri}")
