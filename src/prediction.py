import requests
import argparse
import base64

def parse_args():
    parser = argparse.ArgumentParser(description='Predict sound')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input file')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # mlflow.set_tracking_uri(args.tracking_uri)
    # mlflow.set_experiment(args.experiment_name)
    
    with open(args.input_file, 'rb') as f:
        audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode()
        
    data = {
        'inputs': audio_b64
    }
    
    response = requests.post('http://localhost:5000/invocations', json=data)
    print(response.json())
    
        