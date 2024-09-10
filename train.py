import torch
import dataset as ds
from model import SoundModel
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from utils import save_checkpoint, load_checkpoint
from sklearn.metrics import f1_score
import argparse

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

def train_step(train_loader, val_loader, model, criterion, optimizer, device, num_epochs, prev_epoch=0):
        
        train_losses = []
        train_accuracies = []
        train_f1s = []
        val_losses = []
        val_accuracies = []
        val_f1s = []
        best_f1 = 0
        
        for epoch in tqdm(range(prev_epoch, num_epochs+prev_epoch)):
            
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


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--train_data_dir', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--val_data_dir', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for the dataloader')
    parser.add_argument('--pin_memory', type=bool, default=True, help='Whether to use pin memory for the dataloader')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train the model')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--hidden_size', type=int, default=16, help='Hidden size for the model')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to the model checkpoint')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cpu')
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        
    train_data = ds.get_dataset(args.train_data_dir, transform=None)
    classes = train_data.classes
    val_data = ds.get_dataset(args.val_data_dir, transform=None)
    
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory) # type: ignore
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory) # type: ignore
    
    prev_epoch = 0
    if args.checkpoint is None:
        model = SoundModel(input_shape=1, num_classes=len(classes), hidden_size=args.hidden_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        model, optimizer, prev_epoch, loss, model_f1_score = load_checkpoint(args.checkpoint)
        print(f"Model loaded from epoch {prev_epoch}, with loss: {loss:.4f}, and F1 Score: {model_f1_score:.4f}")
        model = model.to(device)
    
    
    criterion = nn.CrossEntropyLoss()
    train_losses, train_accuracies, val_losses, val_accuracies = train_step(train_loader, val_loader, model, criterion, optimizer, device, num_epochs=args.num_epochs, prev_epoch=prev_epoch)
    
    print("Training Losses: ", train_losses)
    print("Training Accuracies: ", train_accuracies)
    print("Validation Losses: ", val_losses)
    print("Validation Accuracies: ", val_accuracies)