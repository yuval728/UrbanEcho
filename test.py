import torch
import dataset as ds
import torch.nn as nn
from utils import load_checkpoint, load_checkpoint_from_artifact
from sklearn.metrics import f1_score, classification_report
import argparse 
import mlflow

def test(test_dataloader, model, criterion, device):
    
    total_loss = 0
    total_accuracy = 0
    total_f1 = 0
    total_y = []
    total_pred = []
    model.eval()
    
    with torch.inference_mode():
        for i, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)
            output = model(X.unsqueeze(1))
            loss = criterion(output, y)
            total_loss += loss.item()
            accuracy = (output.argmax(1) == y).float().mean()
            total_accuracy += accuracy
            total_f1 += f1_score(y.cpu(), output.argmax(1).cpu(), average='weighted')
            total_y.extend(y.cpu().numpy())
            total_pred.extend(output.argmax(1).cpu().numpy())
            
        total_loss /= len(test_dataloader)
        total_accuracy /= len(test_dataloader)
        total_f1 /= len(test_dataloader)
        
    return total_loss, total_accuracy, total_f1, total_y, total_pred


# def get_registered_model(run_id: str, model_name: str='model', tracking_uri='http://localhost:5000'):
#     mlflow.set_tracking_uri(tracking_uri)
#     model_uri = f"runs:/{run_id}/{model_name}"
#     model = mlflow.pytorch.load_model(model_uri)
#     return model

def parse_args():
    parser = argparse.ArgumentParser(description='Test a model')
    # parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for the dataloader')
    parser.add_argument('--pin_memory', type=bool, default=True, help='Whether to use pin memory for the dataloader')
    parser.add_argument('--run_id', type=str, required=True, help='Run ID of the model')
    parser.add_argument('--artifact_path', type=str, required=True, help='Artifact path of the model')
    
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cpu')
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        
    test_data = ds.get_dataset(args.data_dir, transform=None)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory) # type: ignore
    
    # model, optimizer, epoch, loss, model_f1_score = load_checkpoint(args.checkpoint)
    model,optimizer, epoch, loss, model_f1_score = load_checkpoint_from_artifact(args.run_id, args.artifact_path)
    print(f"Model loaded from epoch {epoch+1}, with loss: {loss:.4f}, and F1 Score: {model_f1_score:.4f}")
    
    criterion = nn.CrossEntropyLoss()
    test_loss, test_accuracy, test_f1, test_y, test_pred = test(test_dataloader, model, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}")
    print(classification_report(test_y, test_pred, target_names=test_data.classes))
    
    
