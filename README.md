# 🧠 FedL-with-FLWR
Federated Learning for Plant Disease Detection (CNN Model)
This repository implements a federated learning system for classifying plant diseases from images using a Convolutional Neural Network (CNN) and the Flower framework. The project is designed to demonstrate how a model can be trained on decentralized data without compromising privacy.

# Key Features
Custom CNN Model: The system uses a custom-built CNN model for image classification.

Federated Averaging (FedAvg): The server aggregates model updates from multiple clients using the FedAvg strategy.

Decentralized Training: The training is distributed across two clients, each using its own local data.

Performance Metrics: The system tracks and logs accuracy and loss metrics, which are saved in a CSV file and visualized in a graph.

# Requirements
The project requires the following Python libraries. You can install them using pip:

pip install flwr==1.17.0 torch==2.5.1 pandas scikit-learn

# Data Setup
The project expects the dataset to be organized in a specific directory structure. You can use the data_split.py script from our discussion to prepare your data.

data/
  ├── Client1/
  │   ├── train/
  │   └── test/
  └── Client2/
      ├── train/
      └── test/
# How to Run
To run the federated learning simulation, follow these steps in three separate terminals:

Start the Server:
python server.py

Start Client 1:
python client_1.py

Start Client 2:
python client_2.py
