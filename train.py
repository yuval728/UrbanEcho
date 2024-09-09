import torch
import dataset as ds
from model import SoundModel
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from utils import save_checkpoint
from sklearn.metrics import f1_score

def train(train_loader, model, criterion, optimizer, device):
    
    total_loss = 0
    total_accuracy = 0
    total_f1 = 0
    model.train()
    
    for i, (input_batch, target_batch) in enumerate(train_loader):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        optimizer.zero_grad()
        output = model(input_batch.unsqueeze(1))
        loss = criterion(output, target_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        accuracy = (output.argmax(1) == target_batch).float().mean()
        total_accuracy += accuracy
        total_f1 += f1_score(target_batch.cpu(), output.argmax(1).cpu(), average='weighted')
         
    total_loss /= len(train_loader)
    total_accuracy /= len(train_loader)
    total_f1 /= len(train_loader)
    return total_loss, total_accuracy, total_f1

def evaluate(test_loader, model, criterion, device):
    
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

def train_step(train_loader, val_loader, model, criterion, optimizer, device, num_epochs):
        
        train_losses = []
        train_accuracies = []
        train_f1s = []
        val_losses = []
        val_accuracies = []
        val_f1s = []
        best_f1 = 0
        
        for epoch in tqdm(range(num_epochs)):
            
            train_loss, train_accuracy, train_f1 = train(train_loader, model, criterion, optimizer, device)
            val_loss, val_accuracy, val_f1 = evaluate(val_loader, model, criterion, device)
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                save_checkpoint(model, optimizer, epoch, val_loss, val_f1, 'checkpoints', is_best=True)
            else:
                save_checkpoint(model, optimizer, epoch, val_loss, val_f1, 'checkpoints', is_best=False)
            
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            train_f1s.append(train_f1)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            val_f1s.append(val_f1)
            
            print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
            
        return train_losses, train_accuracies, val_losses, val_accuracies

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader, classes = ds.get_dataloader('Signals/train', batch_size=128, shuffle=True, num_workers=4, pin_memory=True, transform=None)
    val_loader, _ = ds.get_dataloader('Signals/val', batch_size=128, shuffle=False, num_workers=4, pin_memory=True, transform=None)
    
    model = SoundModel(input_shape=1, num_classes=len(classes), hidden_size=16).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses, train_accuracies, val_losses, val_accuracies = train_step(train_loader, val_loader, model, criterion, optimizer, device, num_epochs=1)
    
    print("Training Losses: ", train_losses)
    print("Training Accuracies: ", train_accuracies)
    print("Validation Losses: ", val_losses)
    print("Validation Accuracies: ", val_accuracies)