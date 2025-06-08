import argparse
import os
from typing import List
from typing import Optional
import time
from threading import Thread
import yaml
import pickle

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

# Import logging service
from LOGGING_SERVICE.logger import get_logger, shutdown_logging

# Import monitoring
from MONITORING_SERVICE.monitoring import (
    GPUMetricsManager,
    ModelMetricsManager,
    TrainingMetricsManager
)

# Import shared model
from model.model import Net

# For metrics server
from prometheus_client import start_http_server

# Initialize training loggers
stdout = get_logger('stdout', 'training')
stderr = get_logger('stderr', 'training')
syslog = get_logger('syslog', 'training')
app_logger = get_logger('app', 'training')

# Initialize the loggers
if not all([stdout, stderr, syslog, app_logger]):
    raise RuntimeError("Failed to initialize loggers")

# Initialize metrics managers
gpu_metrics = GPUMetricsManager(service_name="training")
model_metrics = ModelMetricsManager(service_name="training")
training_metrics = TrainingMetricsManager()

# Load config
def load_config():
    with open('train_config.yml', 'r') as f:
        config = yaml.safe_load(f)
    return config

CONFIG = load_config()
BATCHSIZE = CONFIG['training']['batch_size']
CLASSES = CONFIG['training']['classes']
EPOCHS = CONFIG['training']['epochs']
TRIALS = CONFIG['training']['trials']
TIMEOUT = CONFIG['training']['timeout']
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
        
        # Start timing
        cpu_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_start = time.time()
            
        output = self(data)
        loss = F.nll_loss(output, target)
        
        # Record timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_time = time.time() - gpu_start
        else:
            gpu_time = 0
        cpu_time = time.time() - cpu_start
        
        # Update metrics
        gpu_metrics.update_gpu_metrics()
        model_metrics.record_inference_metrics(gpu_time, cpu_time)
        training_metrics.record_training_metrics(loss=loss.item())
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        data, target = batch
        output = self(data)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(target.view_as(pred)).float().mean()
        
        # Record metrics
        training_metrics.record_training_metrics(accuracy=accuracy.item())
        
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
        optimizer = optim.Adam(self.model.parameters())
        # Record learning rate
        training_metrics.record_training_metrics(lr=optimizer.param_groups[0]['lr'])
        return optimizer

    def on_train_epoch_end(self):
        # Calculate progress percentage
        current_epoch = self.current_epoch + 1
        total_epochs = self.trainer.max_epochs
        progress = (current_epoch / total_epochs) * 100
        
        return super().on_train_epoch_end()


class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        # augmentation based on config
        transforms_list = [transforms.ToTensor()]
        if CONFIG['data']['augmentation']['use_horizontal_flip']:
            transforms_list.insert(0, transforms.RandomHorizontalFlip())
        if CONFIG['data']['augmentation']['random_rotation']:
            transforms_list.insert(0, transforms.RandomRotation(CONFIG['data']['augmentation']['random_rotation']))
        
        self.train_transform = transforms.Compose(transforms_list)
        self.test_transform = transforms.ToTensor()

    def setup(self, stage: Optional[str] = None) -> None:
        train_size, val_size = CONFIG['data']['train_val_split']

        self.mnist_test = datasets.FashionMNIST(
            self.data_dir, train=False, download=True, transform=self.test_transform
        )
        mnist_full = datasets.FashionMNIST(
            self.data_dir, train=True, download=True, transform=self.train_transform
        )
        self.mnist_train, self.mnist_val = random_split(mnist_full, [train_size, val_size])

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
    try:
        stdout.info("")
        stdout.info(f"Starting trial {trial._trial_id}")

        # Set random seed for reproducibility
        pl.seed_everything(42)

        # We optimize the number of layers, hidden units in each layer and dropouts.
        n_layers = trial.suggest_int("n_layers", CONFIG['model']['min_layers'], CONFIG['model']['max_layers'])
        dropout = trial.suggest_float("dropout", CONFIG['model']['dropout_min'], CONFIG['model']['dropout_max'])
        output_dims = [
            trial.suggest_int("n_units_l{}".format(i), CONFIG['model']['min_units'], CONFIG['model']['max_units'], log=True) 
            for i in range(n_layers)
        ]

        stdout.info(f"Trial {trial._trial_id} parameters: layers={n_layers}, dropout={dropout}, dims={output_dims}")

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
        
        mlflow_uri = "http://mlflow:5000"
        experiment_name = "Pytorch_Optuna"

        mlflow.set_tracking_uri(mlflow_uri)
        # Set artifact location to be relative to WORKDIR
        os.environ["MLFLOW_ARTIFACT_ROOT"] = "/app/mlartifacts"
        stdout.info(f"Setting MLflow artifact root to: {os.environ['MLFLOW_ARTIFACT_ROOT']}")
        stdout.info(f"Current working directory: {os.getcwd()}")
        
        mlflow.set_experiment(experiment_name)

        mlflow_logger = MLFlowLogger(
            experiment_name=experiment_name,
            tracking_uri=mlflow_uri,
            run_name=f"Trial:{trial_runid}_{datetime.now().strftime('%d/%m/%Y_%H:%M')}"
        )
        stdout.info(f"MLflow logger configured")

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
            mlflow_runid = run.info.run_id
            stdout.info(f"MLflow run started with ID: {mlflow_runid}")

            try:
                # Log hyperparameters
                hyperparameters = dict(n_layers=n_layers, dropout=dropout, output_dims=output_dims)
                trainer.logger.log_hyperparams(hyperparameters)
                
                # Train the model
                trainer.fit(model, datamodule=datamodule)
                stdout.info(f"Training completed for trial {trial_runid}")

                mlflow.set_tag("mlflow.runName", f"trial:{trial_runid}_Summary")

                # Load the best checkpoint into the model (if available)
                if checkpoint_callback.best_model_path:
                    ckpt_path = checkpoint_callback.best_model_path
                    stdout.info(f"BEST CHECKPOINT PATH OF TRIAL {trial_runid}: {ckpt_path}")

                    # Load and prepare model for saving
                    stdout.info(f"Loading best model from checkpoint...")
                    modelBest = LightningNet.load_from_checkpoint(ckpt_path, dropout=dropout, output_dims=output_dims)
                    modelBest.eval()  # Set to evaluation mode
                    modelBest = modelBest.cpu()  # Move to CPU before saving

                    # Log model with explicit artifact path
                    artifact_path = f"Trial_{trial_runid}_BestModel"
                    stdout.info(f"Attempting to save model with MLflow...")
                    stdout.info(f"Artifact path: {artifact_path}")
                    stdout.info(f"Model will be registered as: fashion_mnist_trial_{trial_runid}")
                    
                    # Debug directory permissions and existence
                    artifact_root = os.environ.get('MLFLOW_ARTIFACT_ROOT', '/app/mlartifacts')
                    stdout.info(f"Checking artifact root directory: {artifact_root}")
                    if os.path.exists(artifact_root):
                        stdout.info(f"Artifact root exists with permissions: {oct(os.stat(artifact_root).st_mode)[-3:]}")
                    else:
                        stdout.error(f"Artifact root directory does not exist!")
                        os.makedirs(artifact_root, exist_ok=True, mode=0o777)
                        stdout.info(f"Created artifact root with full permissions")
                    
                    # Ensure artifact path is relative to WORKDIR
                    local_artifact_path = os.path.join(artifact_root,
                                                     str(mlflow.active_run().info.experiment_id),
                                                     mlflow_runid,
                                                     'artifacts',
                                                    )
                    os.makedirs(local_artifact_path, exist_ok=True, mode=0o777)
                    stdout.info(f"Created local artifact directory: {local_artifact_path}")
                    
                    # Save model directly using save_model
                    mlflow.pytorch.save_model(
                        pytorch_model=modelBest.model,  # Save only the inner model
                        path=local_artifact_path,
                        code_paths=['model/model.py']  # Include the model definition file
                    )
                    stdout.info("Model saved successfully with MLflow")
                    
                    # Test best model      
                    test_result = trainer.test(modelBest, datamodule=datamodule, verbose=False)
                    test_acc = test_result[0].get("test_acc", None)
                    if test_acc is not None:
                        stdout.info(f"Test accuracy: {test_acc}")
                        mlflow.log_metric("test_acc", test_acc)

                # Evaluate on validation set
                val_result = trainer.validate(modelBest, datamodule=datamodule, verbose=False)
                val_acc = val_result[0].get("val_acc", None)
                if val_acc is None:
                    raise ValueError("val_acc not found in validation metrics")
                
                stdout.info(f"Trial {trial_runid} completed with validation accuracy: {val_acc}")
                return val_acc

            except Exception as e:
                stderr.error(f"Error in trial {trial_runid}: {str(e)}", exc_info=True)
                syslog.error(f"Training failure in trial {trial_runid}")
                raise

    except Exception as e:
        stderr.error(f"Trial setup failed: {str(e)}", exc_info=True)
        raise

def main():
    try:
        parser = argparse.ArgumentParser(description="PyTorch Lightning example.")
        parser.add_argument(
            "--pruning",
            "-p",
            action="store_true",
            help="Activate the pruning feature. `MedianPruner` stops unpromising "
            "trials at the early stages of training.",
        )
        args = parser.parse_args()

        stdout.info("-"*50)        
        stdout.info("Starting optimization process")
        syslog.info("Training server started")

        # Start Prometheus metrics server on port 8002 (different from FastAPI)
        start_http_server(8002)
        stdout.info("Metrics server started on port 8002")

        # Create directories if they don't exist
        os.makedirs("./data", exist_ok=True)
        os.makedirs("./checkpoints", exist_ok=True)

        pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()

        study = optuna.create_study(direction="maximize", pruner=pruner)
        study.optimize(objective, n_trials=TRIALS, timeout=TIMEOUT)

        # Log results
        stdout.info(" ")
        stdout.info("*"*30)
        stdout.info("Optimization completed")
        stdout.info(f"Number of finished trials: {len(study.trials)}")
        trial = study.best_trial
        stdout.info(f"Best trial: trial {trial._trial_id}")
        stdout.info(f"Validation accuracy: {trial.value:.4f}")
        
        stdout.info("Best parameters:")
        for key, value in trial.params.items():
            stdout.info(f"\t{key}: {value:.4f}")
        
        stdout.info("*"*30)
        stdout.info("-"*50)
        stdout.info("\n")

    except Exception as e:
        stderr.error("Fatal error in optimization process", exc_info=True)
        syslog.error("Training server crashed")
    
    finally:
        # Cleanup
        shutdown_logging('training')

if __name__ == "__main__":
    main()


# mlflow server --host localhost --port 8080
