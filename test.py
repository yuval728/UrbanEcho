import torch
import dataset as ds
from model import SoundModel
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from utils import load_checkpoint
from sklearn.metrics import f1_score

def test(test_loader, model, criterion, device):
    
    total_loss = 0
    total_accuracy = 0
    total_f1 = 0
    model.eval()
    
    with torch.inference_mode():
        for i, (input_batch, target_batch) in enumerate(test_loader):
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            output = model(input_batch.unsqueeze(1))
            loss = criterion(output, target_batch)
            total_loss += loss.item()
            accuracy = (output.argmax(1) == target_batch).float().mean()
            total_accuracy += accuracy
            total_f1 += f1_score(target_batch.cpu(), output.argmax(1).cpu(), average='weighted')
            
        total_loss /= len(test_loader)
        total_accuracy /= len(test_loader)
        total_f1 /= len(test_loader)
    return total_loss, total_accuracy, total_f1

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_loader, classes = ds.get_dataloader('Signals/test', batch_size=32, shuffle=False,num_workers=4, pin_memory=True, transform=None)
    
    criterion = nn.CrossEntropyLoss()
    model, optimizer, epoch, loss, model_f1_score = load_checkpoint( 'checkpoints/checkpoint.pth.tar')
    print(f"Model loaded from epoch {epoch}, with loss: {loss:.4f}, and F1 Score: {model_f1_score:.4f}")
    test_loss, test_accuracy, test_f1 = test(test_loader, model, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}")
    
