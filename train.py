import torch
import dataset as ds
from model import SoundModel
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from utils import save_checkpoint, load_checkpoint, create_signature_log_model
from sklearn.metrics import f1_score
import argparse
import mlflow

torch.backends.cudnn.benchmark = True


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
        total_f1 += f1_score(
            target_batch.cpu(), output.argmax(1).cpu(), average="weighted"
        )

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
            total_f1 += f1_score(
                target_batch.cpu(), output.argmax(1).cpu(), average="weighted"
            )

        total_loss /= len(test_loader)
        total_accuracy /= len(test_loader)
        total_f1 /= len(test_loader)
    return total_loss, total_accuracy, total_f1


def train_step(
    train_loader,
    val_loader,
    model,
    criterion,
    optimizer,
    device,
    num_epochs,
    run_name,
    prev_epoch=0,
):
    best_val_f1 = 0

    for epoch in tqdm(range(prev_epoch, num_epochs + prev_epoch)):
        train_loss, train_accuracy, train_f1 = train(
            train_loader, model, criterion, optimizer, device
        )
        val_loss, val_accuracy, val_f1 = evaluate(val_loader, model, criterion, device)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

            save_checkpoint(
                model, optimizer, epoch, val_loss, val_f1, "checkpoints", is_best=True
            )
        else:
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_f1, "checkpoints", is_best=False
            )

        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_f1": val_f1,
            },  # type: ignore
            step=epoch,
        )

        print(
            f"Epoch: {epoch+1}/{num_epochs+prev_epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}"
        )

    create_signature_log_model(model, device)

    return train_loss, train_accuracy, train_f1, val_loss, val_accuracy, val_f1


def load_model_optimizer(args, classes, device):
    prev_epoch = 0
    
    if args.checkpoint is None:
        model = SoundModel(
            input_shape=1, num_classes=len(classes), hidden_size=args.hidden_size
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        model, optimizer, prev_epoch, loss, model_f1_score = load_checkpoint(
            args.checkpoint
        )
        print(
            f"Model loaded from epoch {prev_epoch}, with loss: {loss:.4f}, and F1 Score: {model_f1_score:.4f}"
        )
        model = model.to(device)

    return model, optimizer, prev_epoch


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--train_data_dir", type=str, required=True, help="Path to the data directory"
    )
    parser.add_argument(
        "--val_data_dir", type=str, required=True, help="Path to the data directory"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use for training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for the dataloader",
    )
    parser.add_argument(
        "--pin_memory",
        type=bool,
        default=True,
        help="Whether to use pin memory for the dataloader",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of epochs to train the model"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=16, help="Hidden size for the model"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="SoundClassification",
        help="Name of the Mlflow experiment",
    )
    parser.add_argument(
        "--mlflow_uri",
        type=str,
        default="http://127.0.0.1:5000/",
        help="URI of the MLFlow server",
    )
    # parser.add_argument(
    #     "--run_id", type=str, default=None, help="id of the MLFlow run to resume"
    # )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cpu")
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")

    ##Setting up MLFlow
    mlflow.set_experiment(args.experiment_name)
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.start_run(log_system_metrics=True, nested=False)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_data = ds.get_dataset(args.train_data_dir, transform=None)
    classes = train_data.classes
    val_data = ds.get_dataset(args.val_data_dir, transform=None)

    train_loader = torch.utils.data.DataLoader(  # type: ignore
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )  # type: ignore
    val_loader = torch.utils.data.DataLoader(  # type: ignore
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )  # type: ignore

    mlflow.log_param("num_classes", len(classes))
    mlflow.log_params(vars(args))
    # mlflow.log_param('classes', classes)

    model, optimizer, prev_epoch = load_model_optimizer(args, classes, device)
    mlflow.log_param('prev_epoch', prev_epoch)
    

    criterion = nn.CrossEntropyLoss()

    train_loss, train_accuracy, train_f1, val_loss, val_accuracy, val_f1 = train_step(
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        device,
        args.num_epochs,
        args.experiment_name,
        prev_epoch,
    )

    print(
        f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}"
    )

    mlflow.end_run()
