import mlflow
import logging
import yaml
import torch
from torch.utils.data import DataLoader
from src.create_input_files import create_input_files
from src.dataset import get_dataset
from src.model import SoundModel
from src.train import train_step
from src.test import test
from src.utils import load_checkpoint
from src.model_registry import (
    UrbanEcho,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


def main():
    # Load configuration
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(log_system_metrics=True) as run:
        mlflow.log_params(config)

        # Create input files
        logging.info("Creating input files...")
        # create_input_files(
        #     csv_file=config["csv_file"],
        #     data_dir=config["data_dir"],
        #     input_data_dir=config["input_data_dir"],
        #     n_mfcc=config["n_mfcc"],
        #     test_size=config["test_size"],
        #     val_size=config["val_size"],
        #     seed=config["seed"],
        # )
        logging.info("Input files created successfully!")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Device: {device}")

        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])

        # Load datasets
        logging.info("Loading datasets...")
        train_data = get_dataset(config["train_data_dir"], transform=None)
        classes = train_data.classes
        val_data = get_dataset(config["val_data_dir"], transform=None)
        logging.info("Datasets loaded successfully!")

        # Create data loaders
        logging.info("Creating data loaders...")
        train_loader = DataLoader(
            train_data,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
        )

        val_loader = DataLoader(
            val_data,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
        )
        logging.info("Data loaders created successfully!")

        # Load model
        logging.info("Loading model, optimizer and previous epoch...")
        if config["checkpoint"] is not None:
            model, optimizer, prev_epoch, _, _ = load_checkpoint(config["checkpoint"])
            model = model.to(device)
        else:
            model = SoundModel(
                input_shape=1,
                num_classes=len(classes),
                hidden_size=config["hidden_size"],
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
            prev_epoch = 0
        logging.info("Model, optimizer and previous epoch loaded successfully!")

        criterion = torch.nn.CrossEntropyLoss()

        # Train model
        logging.info("Training model...")
        train_loss, train_accuracy, train_f1, val_loss, val_accuracy, val_f1 = (
            train_step(
                train_loader,
                val_loader,
                model,
                criterion,
                optimizer,
                device,
                config["num_epochs"],
                config["checkpoint_save_path"],
                prev_epoch,
            )
        )
        logging.info(
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}"
        )

        # Test model
        logging.info("Testing model...")
        test_data = get_dataset(config["test_data_dir"], transform=None)
        test_loader = DataLoader(
            test_data,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
        )

        if config["checkpoint"] is not None:
            model, _, _, _, _ = load_checkpoint(config["checkpoint"])
        test_loss, test_accuracy, test_f1, test_y, test_pred = test(
            test_loader, model, criterion, device
        )

        logging.info(
            f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}"
        )

        # Register model
        logging.info("Registering model...")
        sampled_input = torch.rand(1, *[1, config["n_mfcc"]])
        traced_model = torch.jit.trace(model, sampled_input)
        traced_model_path = f"{config['model_name']}.pt"
        torch.jit.save(traced_model, traced_model_path)
        mlflow.pyfunc.log_model(
            artifact_path=config["model_name"],
            python_model=UrbanEcho(n_mfcc=config["n_mfcc"], classes=classes),
            artifacts={"model": traced_model_path},
        )

        mlflow.register_model(
            f"runs:/{run.info.run_id}/{config['model_name']}", config["model_name"]
        )

        logging.info("Model registered successfully!")

        logging.info("End of training!")


if __name__ == "__main__":
    main()
