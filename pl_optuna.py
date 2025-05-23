import argparse
import os
from typing import List
from typing import Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from pytorch_lightning.loggers import MLFlowLogger

from datetime import datetime
import mlflow
import mlflow.pytorch

import optuna
from optuna.integration import PyTorchLightningPruningCallback

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms


BATCHSIZE = 128
CLASSES = 10
EPOCHS = 2
TRIALS = 3
DIR = os.getcwd()

# python 3.10.16

class Net(nn.Module):
    def __init__(self, dropout: float, output_dims: List[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []

        input_dim: int = 28 * 28
        for output_dim in output_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = output_dim

        layers.append(nn.Linear(input_dim, CLASSES))

        self.layers = nn.Sequential(*layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        logits = self.layers(data)
        return F.log_softmax(logits, dim=1)


class LightningNet(pl.LightningModule):
    def __init__(self, dropout: float, output_dims: List[int]) -> None:
        super().__init__()
        self.model = Net(dropout, output_dims)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data.view(-1, 28 * 28))

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        data, target = batch
        output = self(data)
        loss = F.nll_loss(output, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return F.nll_loss(output, target)

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        data, target = batch
        output = self(data)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(target.view_as(pred)).float().mean()
        self.val_accuracy = accuracy
        self.log("val_acc", accuracy)
        self.log("hp_metric", accuracy, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        data, target = batch
        output = self(data)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(target.view_as(pred)).float().mean()
        self.log("test_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.model.parameters())


class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size


        # augmentation
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor()
        ])
        
        self.test_transform = transforms.ToTensor()

    def setup(self, stage: Optional[str] = None) -> None:

        self.mnist_test = datasets.FashionMNIST(
            self.data_dir, train=False, download=True, transform=self.test_transform
        )
        mnist_full = datasets.FashionMNIST(
            self.data_dir, train=True, download=True, transform=self.train_transform
        )
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Apply test transform to validation set
        self.mnist_val.dataset.transform = self.test_transform

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            shuffle=False,
            
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            shuffle=False,
        )


def objective(trial: optuna.trial.Trial) -> float:

    # Set random seed for reproducibility
    pl.seed_everything(42)

    # We optimize the number of layers, hidden units in each layer and dropouts.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    output_dims = [
        trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True) for i in range(n_layers)
    ]

    # Use a trial-specific checkpoint directory
    trial_runid = trial._trial_id
    checkpoint_dir = os.path.join(DIR, "checkpoints", f"trial_{trial_runid}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    dataset_info = {
        "dataset_name": "FashionMNIST",
        "source": "torchvision.datasets.FashionMNIST",
        "version": torch.__version__,
        "link": "https://github.com/zalandoresearch/fashion-mnist",
        "batch_size": BATCHSIZE,
        "train_samples": 55000,
        "val_samples": 5000,
        "test_samples": 10000
    }
    
    mlflow_uri = "http://localhost:8080"
    experiment_name = "Pytorch_Optuna"

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)


    mlflow_logger = MLFlowLogger( # Pytoch Lightning
        experiment_name=experiment_name,
        tracking_uri=mlflow_uri,
        run_name=f"Trial:{trial_runid}_{datetime.now().strftime('%d/%m/%Y_%H:%M')}"
    )

    # Save checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath=checkpoint_dir,
        filename=f"{trial_runid}-" + 'epoch={epoch}-val_acc={val_acc:.2f}',
        save_top_k=3,
        mode='max'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        patience=3,
        mode='max'
    )

    model = LightningNet(dropout, output_dims)
    datamodule = FashionMNISTDataModule(data_dir=f"{DIR}/data", batch_size=BATCHSIZE)


    trainer = pl.Trainer(
        logger=mlflow_logger,
        enable_checkpointing=True,
        max_epochs=EPOCHS,
        accelerator="auto",
        devices=1,
        callbacks=[early_stop_callback, PyTorchLightningPruningCallback(trial, monitor="val_acc"), checkpoint_callback],
    )

 
    # Start MLflow run first
    with mlflow.start_run(nested=True) as run:

        # for mlflow
        mlflow_runid = run.info.run_id
        

        # Log hyperparameters
        hyperparameters = dict(n_layers=n_layers, dropout=dropout, output_dims=output_dims)
        trainer.logger.log_hyperparams(hyperparameters)
        
        # Train the model
        trainer.fit(model, datamodule=datamodule)

        mlflow.set_tag("mlflow.runName", f"trial:{trial_runid}_Summary")

        # Load the best checkpoint into the model (if available)
        if checkpoint_callback.best_model_path:
            ckpt_path = checkpoint_callback.best_model_path
            print(f"BEST CHECKPOINT PATH OF TRIAL {trial_runid}: {ckpt_path}")

            
            modelBest = LightningNet.load_from_checkpoint(ckpt_path, dropout=dropout, output_dims=output_dims)
            mlflow.pytorch.log_model(modelBest, f"Trial_{trial_runid}_BestModel")
            print("BEST MODEL LOADED")

            mlflow.log_param("best_checkpoint_path", ckpt_path)

            for i, path in enumerate(checkpoint_callback.best_k_models.keys()):
                mlflow.log_param(f"checkpoint_{i}_path", path)
                mlflow.log_param(f"checkpoint_{i}_val_acc", checkpoint_callback.best_k_models[path].item())

            mlflow.log_artifacts(checkpoint_dir, artifact_path=f"Trial_{trial_runid}_ckpt")

            # Test best model      
            test_result = trainer.test(modelBest, datamodule=datamodule, verbose=False)
            test_acc = test_result[0].get("test_acc", None)
            if test_acc is not None:
                mlflow.log_metric("test_acc", test_acc)

        # Evaluate on validation set
        val_result = trainer.validate(modelBest, datamodule=datamodule, verbose=False)
        val_acc = val_result[0].get("val_acc", None)
        if val_acc is None:
            raise ValueError("val_acc not found in validation metrics")
        
    return val_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Lightning example.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    args = parser.parse_args()

    # Create directories if they don't exist
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)

    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=TRIALS, timeout=600)

    print("\n\n")
    print("*"*20)
    print("Number of finished trials: {}".format(len(study.trials)))
    
    trial = study.best_trial
    print(f"Best trial: trial {trial._trial_id}")

    print("\tValidation accuracy: {:.4f}".format(trial.value))

    print("\tParams: ")
    for key, value in trial.params.items():
        print("\t\t{}: {:.4f}".format(key, value))
    
    print("*"*20)


# mlflow server --host localhost --port 8080
