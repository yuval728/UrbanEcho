import torch
import mlflow
import os
import utils as ut
import argparse

# def feature_extraction(file_name, n_mfcc=50):
#     sample,sample_rate = librosa.load(file_name,res_type='kaiser_fast')
#     feature = librosa.feature.mfcc(y=sample,sr=sample_rate,n_mfcc=n_mfcc)
#     scaled_feature = np.mean(feature.T,axis=0)
#     return scaled_feature


def prediction_of_sound(file_name, model, n_mfcc=50):
    scaled_feature = ut.feature_extraction(file_name, n_mfcc)
    scaled_feature = torch.tensor(scaled_feature).float()
    scaled_feature = scaled_feature.unsqueeze(0)
    model.eval()
    with torch.inference_mode():
        output = model(scaled_feature.unsqueeze(1))
        prediction = torch.nn.functional.softmax(output, dim=1)
    return prediction

def parse_args():
    parser = argparse.ArgumentParser(description='Predict sound')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input file')
    parser.add_argument('--n_mfcc', type=int, default=50, help='Number of MFCC features')
    parser.add_argument('--model_name', type=str, default='model', help='Name of the model')
    parser.add_argument('--model_version', type=int, default=1, help='Version of the model')
    parser.add_argument('--tracking_uri', type=str, default='http://localhost:5000', help='URI of the tracking server')
    parser.add_argument('--experiment_name', type=str, default='Default', help='Name of the experiment')
    parser.add_argument('--classes', type=str, default='classes.txt', help='Path to the classes file')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # mlflow.set_tracking_uri(args.tracking_uri)
    # mlflow.set_experiment(args.experiment_name)
    
    model = mlflow.pytorch.load_model(f"models:/{args.model_name}/{args.model_version}")
    
    prediction = prediction_of_sound(args.input_file, model, args.n_mfcc)
    
    print('Prediction:', prediction)
    
    with open(args.classes, 'r') as f:
        classes = f.readlines()
        print('Class:', classes[prediction.argmax().item()])
        