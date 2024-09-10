import torch
import dataset as ds
import torch.nn as nn
from utils import load_checkpoint
from sklearn.metrics import f1_score
import argparse 

def test(test_dataloader, model, criterion, device):
    
    total_loss = 0
    total_accuracy = 0
    total_f1 = 0
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
            
        total_loss /= len(test_dataloader)
        total_accuracy /= len(test_dataloader)
        total_f1 /= len(test_dataloader)
    return total_loss, total_accuracy, total_f1

def parse_args():
    parser = argparse.ArgumentParser(description='Test a model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for the dataloader')
    parser.add_argument('--pin_memory', type=bool, default=True, help='Whether to use pin memory for the dataloader')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cpu')
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        
    test_data = ds.get_dataset(args.data_dir, transform=None)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory) # type: ignore
    
    model, optimizer, epoch, loss, model_f1_score = load_checkpoint(args.checkpoint)
    print(f"Model loaded from epoch {epoch}, with loss: {loss:.4f}, and F1 Score: {model_f1_score:.4f}")
    
    criterion = nn.CrossEntropyLoss()
    test_loss, test_accuracy, test_f1 = test(test_dataloader, model, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}")
    
