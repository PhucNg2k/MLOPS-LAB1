# 🚀 PyTorch Lightning with Optuna and MLflow

<div align="center"> 
<img src="https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png" alt="PyTorch" height="60"/> &nbsp;&nbsp;&nbsp; <img src="https://raw.githubusercontent.com/Lightning-AI/lightning/master/docs/source-pytorch/_static/images/logo.png" alt="PyTorch Lightning" height="60"/> &nbsp;&nbsp;&nbsp; <img src="https://raw.githubusercontent.com/optuna/optuna/master/docs/image/optuna-logo.png" alt="Optuna" height="60"/> &nbsp;&nbsp;&nbsp;  


![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
</div>

## 👤 Student Information

| Full Name           | Student ID |
|---------------------|-----------|
| Nguyen Thuong Phuc  | 22521134  |

---

## 🔧 Pipeline Overview

This project builds a **model training pipeline** for **image classification** using:
- **Optuna** for automatic hyperparameter optimization.
- **MLflow** for experiment tracking (logs, checkpoints, hyperparameters,...).
- **PyTorch Lightning** to simplify training loops and organize code clearly.

The pipeline operates completely automatically:
1. Train neural networks with the FashionMnist dataset
2. **Create an Optuna study** to run multiple training trials with different hyperparameters, optimizing for validation accuracy.
3. **MLflow** records all information for each trial: model, val/test accuracy, checkpoints,...
4. Automatically stop inefficient trials early with `EarlyStopping` and `PruningCallback`.
5. Automatically save and reload checkpoints with the best validation accuracy.

> 🔥 **Innovations / creative points**:  
> - Fully integrates all 3 modern tools: PytorchLightning + Optuna + MLflow.
> - Restructured pipeline for easy expansion, easy management of logs and models.
> - Can run with just one command line (`python pl_optuna.py -p`), everything else is automated.

---

## 🧠 Technologies Used and Key Features

| Technology          | Role                                                       |
|-------------------|---------------------------------------------------------------|
| **PyTorch Lightning** | Organize clean training loops, support for callbacks, automatic logging |
| **Optuna**         | Hyperparameter optimization with pruning                          |
| **MLflow**         | Log experiments, checkpoints, models, metrics                |
| **FashionMNIST**   | Demo dataset (28x28 fashion images)                      |

---

## Installation Guide

1. Clone the project from GitHub:
   ```bash
   git clone https://github.com/PhucNg2k/MLOPS-LAB1.git
   cd MLOPS-LAB1
   ```
2. Install the necessary libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Guide

**Step 1: Start the MLflow Server**

Open a terminal and run the following command to start the MLflow server:

```bash
mlflow server --host localhost --port 8080
```

The MLflow interface will be available at http://localhost:8080.

**Step 2: Run the training script**

Open another terminal and run the training script with pruning enabled:

```bash
python pl_optuna.py -p
```

## Results

The results of each trial will be displayed in the terminal (accuracy, params,...).

MLflow UI will store:

+ Hyperparameters, metrics
+ Best model checkpoints
+ Logs, artifacts

Run names will be set with readable identifiers (E.g.: Trial:0_14/04/2025_18:32)

## Notes
Make sure the mlflow server is started before running the training script.

You can modify the pl_optuna.py file to customize experiments according to your needs.

## 🎥 Video Demo

[![Demo Video](https://img.youtube.com/vi/mela8dFpKq0/0.jpg)](https://www.youtube.com/watch?v=mela8dFpKq0)