import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
from modelCNN import CNN, traintest

def load_data(train_path='data/Client1/train', test_path='data/Client1/test'):
    """Loads both the train and test datasets and returns their sizes."""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
    return len(train_dataset), len(test_dataset)


def analyze_cnn_model(model_path="trained_model_client_1.pt", test_path='data/Client1/test'):
    """
    Analyzes the CNN model by checking dataset size,
    evaluating performance, and displaying the metric matrix.
    """
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}. Please run the federated training first.")
        return

    # --- 1. Check Dataset Size ---
    print("--- Dataset Information ---")
    train_size, test_size = load_data(test_path=test_path)
    total_size = train_size + test_size
    print(f"Train dataset size: {train_size} images")
    print(f"Test dataset size: {test_size} images")
    print(f"Total dataset size: {total_size} images\n")


    # --- 2. Evaluate Trained Model ---
    print("--- Model Performance on Test Set ---")
    # This part of the code is the same as before
    # ...


    # --- 3. Display Metric Matrix ---
    print("--- Metric Matrix (from metrics_log.csv) ---")
    log_file = "metrics_log.csv"
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        print(df.to_markdown(index=False))
    else:
        print("Error: metrics_log.csv not found. Please run the federated training to generate it.")

if __name__ == "__main__":
    analyze_cnn_model()
