import os
import torch
import mlflow.pytorch
from torchvision import transforms

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


def load_model(model_dir):
    """
    Load a full model logged with MLflow (LightningModule).
    """
    print(f"Loading model from: {model_dir}")
    model = mlflow.pytorch.load_model(model_dir, map_location=torch.device("cpu"))
    model.eval()
    return model


def inference_model(image_tensor, model, return_label=False):
    """
    Run inference on a preprocessed image tensor using the loaded model.
    image_tensor: torch.Tensor of shape (1, 1, 28, 28)
    Returns: predicted class index (int)
    """
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        if return_label:
            return FASHION_MNIST_CLASSES.get(predicted_class, "Unknown")
        return predicted_class


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
    Given the base folder and a specific run ID, finds the best model path.
    """
    print("folder path:", folder_path)
    print("target run: ", target_run)

    best_model_path = None
    for foldername in os.listdir(folder_path):
        if foldername == target_run:
            run_path = normalize_path(folder_path, target_run)
            best_model_path = search_best_trial_model(run_path)
            break

    if best_model_path is None:
        print("Target run not found or model not found.")
    return best_model_path


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
