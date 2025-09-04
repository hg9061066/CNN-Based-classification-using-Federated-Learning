import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Import the CNN model architecture
from modelCNN import CNN

# Data loading function
def load_test_data(test_path='data/Client1/test'):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    return test_loader, test_dataset.classes

if __name__ == "__main__":
    model_path = "trained_model_client_1.pt"
    
    # Check if the trained model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model file not found: {model_path}. Please run client_1.py to train the model first.")

    # Load the test data and class names
    test_loader, classes = load_test_data()

    # Load the trained CNN model
    model = CNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    y_true = []
    y_pred = []

    print("Running evaluation on test set...")
    # Run the model on the test data to get predictions
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    # Generate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    plt.tight_layout()
    plt.show()

    print("\nConfusion Matrix generated successfully!")