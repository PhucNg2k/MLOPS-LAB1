import os
import torch
import mlflow
from torchvision import transforms
from LOGGING_SERVICE.logger import get_logger
from pathlib import Path
from model.model import Net  # Import from shared model directory

# Initialize loggers
app_logger = get_logger('app', 'api')
sys_logger = get_logger('syslog', 'api')

# Verify loggers are initialized
if not all([app_logger, sys_logger]):
    raise RuntimeError("Failed to initialize API loggers")

FASHION_MNIST_CLASSES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}


def normalize_path(*parts):
    """
    Safely joins and normalizes a path across platforms (Windows, macOS, Linux).
    Example: normalize_path("..", "mlruns", "mlartifacts")
    """
    return os.path.normpath(os.path.join(*parts))


def load_model(model_path):
    """
    Load a PyTorch model from the local MLflow artifacts directory.
    The model_path should point to a directory containing the saved model.
    """
    if not model_path:
        return None
        
    try:
        app_logger.info(f"Loading model from local path: {model_path}")
        
        # Set up MLflow loading context
        mlflow.pytorch.autolog()
        
        # Load model directly
        model = mlflow.pytorch.load_model(
            model_uri=model_path,
            map_location=torch.device('cpu')
        )
        model.eval()
        app_logger.info("Model loaded successfully")
        return model
        
    except Exception as e:
        app_logger.error(f"Failed to load model: {str(e)}")
        return None


def inference_model(image_tensor, model, return_confidence=False):
    """
    Run inference on a preprocessed image tensor using the loaded model.
    image_tensor: torch.Tensor of shape (1, 1, 28, 28)
    Returns: predicted class index (int) and confidence (float) if return_confidence=True
    """
    with torch.no_grad():
        # Reshape from (1, 1, 28, 28) to (1, 784)
        flattened = image_tensor.view(-1, 28 * 28)
        
        output = model(flattened)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        if return_confidence:
            return FASHION_MNIST_CLASSES.get(predicted_class, "Unknown"), confidence
        return FASHION_MNIST_CLASSES.get(predicted_class, "Unknown")


def find_mlrun(folder_path):
    """
    Finds the first folder that looks like a run ID (length between 16-18).
    """
    for foldername in os.listdir(folder_path):
        if 16 <= len(foldername) <= 18:
            return foldername  # Only return the folder name


def search_best_trial_model(run_path):
    """
    Searches for the best trial checkpoint in the given MLflow run path
    based on the highest validation accuracy found in checkpoint filenames.
    """
    val_records = []
    trial_names = []
    idxs_names = []
    for trial in os.listdir(run_path):
        trial_names.append(trial)
        base_path = normalize_path(run_path, trial, "artifacts")
        
        idx_name = 0
        for fol in os.listdir(base_path):
            idx_name = int(fol.split("_")[1])
            break
        idxs_names.append(idx_name)
        ckpt_folder = normalize_path(base_path, f"Trial_{idx_name}_ckpt")

        temp = []
        for ckptfile in os.listdir(ckpt_folder):
            val_acc_str = os.path.splitext(ckptfile.split("=")[-1])[0]
            try:
                val_acc = float(val_acc_str)
                temp.append(val_acc)
            except ValueError:
                print(f"Skipping invalid file: {ckptfile}")
        val_records.append(tuple(temp))

    print("Val records: ", val_records)

    # Find trial with the highest single val_acc
    best_trial_idx = max(enumerate(val_records), key=lambda x: max(x[1]))[0]

    best_model_dir = normalize_path(
        run_path,
        trial_names[best_trial_idx],
        "artifacts",
        f"Trial_{idxs_names[best_trial_idx]}_BestModel"
    )
    return best_model_dir


def get_best_modelFile(folder_path, target_run):
    """
    Find the run with best test_acc in mlruns, then load its model from mlartifacts.
    """
    app_logger.info(f"Looking for best model based on test accuracy")

    try:
        # First check mlruns directory for metrics
        mlruns_dir = normalize_path("mlruns")
        if not os.path.exists(mlruns_dir):
            app_logger.error(f"Mlruns directory not found: {mlruns_dir}")
            return None

        # Get experiment directory (should be a number)
        exp_dirs = [d for d in os.listdir(mlruns_dir) if d != 'models' and d != '.trash']
        if not exp_dirs:
            app_logger.error("No experiment directories found in mlruns")
            return None
        
        exp_dir = exp_dirs[0]  # Use first experiment directory
        exp_path = normalize_path(mlruns_dir, exp_dir)
        
        # Check mlartifacts directory exists
        arti_dir = normalize_path("mlartifacts", exp_dir)
        if not os.path.exists(arti_dir):
            app_logger.error(f"Artifacts directory not found: {arti_dir}")
            return None
        
        # Find best run by checking test_acc in each run
        best_acc = -1
        best_run_id = None
        
        for run_id in os.listdir(exp_path):
            if run_id == 'meta.yaml':  # Skip meta file
                continue
            
            app_logger.info(f"Checking run_id: {run_id}")
            
            # Skip if run doesn't have artifacts
            if run_id not in os.listdir(arti_dir):
                app_logger.info(f"Skipping run {run_id} - no artifacts found")
                continue
                
            # Check if run has test_acc metric
            test_acc_file = normalize_path(exp_path, run_id, "metrics/test_acc")
            if not os.path.exists(test_acc_file):
                app_logger.info(f"Skipping run {run_id} - no test metric found")
                continue
            
            app_logger.info(f"Checking test_acc of {run_id}")

            # Read the last test accuracy value (format: "timestamp value step")
            with open(test_acc_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    # Split by space and take second value (the accuracy)
                    acc = float(lines[-1].strip().split()[1])
                    if acc > best_acc:
                        best_acc = acc
                        best_run_id = run_id
                        app_logger.info(f"New best run {run_id} with test accuracy {acc:.4f}")
        
        if best_run_id is None:
            app_logger.error("Could not find any valid runs with test accuracy")
            return None
            
        app_logger.info(f"Best run is {exp_dir}/{best_run_id} with accuracy {best_acc:.4f}")
        
        # Construct path to model artifacts using run_id
        model_dir = normalize_path("mlartifacts", exp_dir, best_run_id, "artifacts")
        if os.path.exists(model_dir):
            app_logger.info(f"Found best model at: {model_dir}")
            return model_dir
                
        app_logger.error(f"Could not find model for run {best_run_id} in mlartifacts")
        return None

    except Exception as e:
        app_logger.error(f"Error finding best model: {str(e)}")
        return None


def preprocess_image(pil_image):
    """
    Preprocess a PIL image into a normalized tensor suitable for inference.
    Assumes grayscale image of size 28x28.
    """

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    return transform(pil_image).unsqueeze(0)  # Shape: (1, 1, 28, 28)


if __name__ == "__main__":
    # Locate the best model
    folder_root = normalize_path("..", "mlruns")
    target_run = find_mlrun(folder_root)
    folder_root = normalize_path("..", "mlartifacts")
    model_path = get_best_modelFile(folder_root, target_run)

    print("\n✅ Best model path:")
    print(model_path)

    # Load the best model
    model = load_model(model_path)

    # Test inference with a dummy image
    dummy_image = torch.rand(1, 1, 28, 28)  # Simulated image
    pred = inference_model(dummy_image, model)
    print("Predicted class:", pred)
