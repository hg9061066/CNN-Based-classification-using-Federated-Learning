# CNN-Based-classification-using-Federated-Learning
FedL-with-CNN: Federated Learning for Plant Disease Detection. This repository contains a federated learning system for image-based plant disease detection. It uses a custom Convolutional Neural Network (CNN) model built with PyTorch and the Flower (FLWR) framework to train across multiple clients while preserving data privacy.

Model Details
Model Used: 
A custom Convolutional Neural Network (CNN) as defined in your project's code.

Dataset Size: 
The total dataset size is 1124 images on the test set.

Training Configuration: 
The model was trained over 20 global epochs with 5 local epochs per client.

Final Accuracy: 
The model achieved a final accuracy of 99% on the test set, as shown in the confusion matrix.

Final Loss: 
The loss decreased consistently over the training process, reaching a final value of approximately 0.04.

Performance Analysis
The federated learning process was highly successful. The graph of accuracy and loss over rounds shows a clear trend of improvement.

Accuracy: 
The accuracy of the global model starts at approximately 65% and steadily increases, reaching almost 99% by the final round. This indicates that the federated training was effective at improving the model's performance over time.

Loss: 
The loss starts at about 0.12 and consistently decreases, which confirms that the model is learning and converging
