
import numpy as np
import os
import librosa
from tqdm.auto import tqdm
import torch
import mlflow


    
def create_class_folders(base_dir, classes, save_classes=True):
    for class_name in classes:
        path=os.path.join(base_dir, class_name)
        if os.path.exists(path):
            print(f"{path} already exists.")
            continue
        os.makedirs(path,exist_ok=True)
        print('created', path)
    
    if not save_classes:
        return
    
    with open(os.path.join(base_dir, 'classes.txt'), 'w') as f:
        for class_name in classes:
            f.write(class_name+'\n')
        


def feature_extraction(file_name, n_mfcc=50):
    sample,sample_rate = librosa.load(file_name,res_type='kaiser_fast')
    feature = librosa.feature.mfcc(y=sample,sr=sample_rate,n_mfcc=n_mfcc)
    scaled_feature = np.mean(feature.T,axis=0)
    return scaled_feature

def save_features(data_dir, df, output_dir, n_mfcc=50):
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        file_name = os.path.join(os.path.abspath(data_dir),'fold'+str(row["fold"])+'/',str(row['slice_file_name'])) 
        feature = feature_extraction(file_name, n_mfcc)
        np.save(os.path.join(output_dir, row['class'], row['slice_file_name'][:-4]), feature)
        
    print('Features saved to', output_dir)
    

def save_checkpoint(model, optimizer, epoch, loss, f1_score, path, is_best=False):

    os.makedirs(path, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model': model,
        'optimizer': optimizer,
        'loss': loss,
        'f1_score': f1_score
    }
    torch.save(checkpoint, os.path.join(path, 'checkpoint.pth.tar'))
    mlflow.log_artifact(os.path.join(path, 'checkpoint.pth.tar'))
    
    if is_best:
        torch.save(checkpoint, os.path.join(path, 'best.pth.tar'))
        mlflow.log_artifact(os.path.join(path, 'best.pth.tar'))
    # mlflow.pytorch.log_model(model, 'model')
    
    
    print('\nCheckpoint saved to', path)
    
def load_checkpoint(path):
    checkpoint = torch.load(path)
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    f1_score = checkpoint['f1_score']
    
    return model, optimizer, epoch, loss, f1_score

# def create_signature_log_model(model, device):
#     input_sample = torch.randn(1, 1, 50).to(device)
#     signature = mlflow.models.infer_signature(input_sample.detach().numpy(), model(input_sample).detach().numpy())
#     mlflow.pytorch.log_model(model, 'model', signature=signature)
#     print('Model logged to mlflow')
    
def load_checkpoint_from_artifact(run_id: str, artifact_path: str):
    mlflow_client = mlflow.tracking.MlflowClient()
    model_path = mlflow_client.download_artifacts(run_id, artifact_path)
    model, optimizer, epoch, loss, model_f1_score = load_checkpoint(model_path)
    return model, optimizer, epoch, loss, model_f1_score
    