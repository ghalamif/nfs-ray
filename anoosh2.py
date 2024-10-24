import os
import ray.data as rd
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from ray.air import session
from ray.train.torch import TorchTrainer
import ray
from ray.train import ScalingConfig

# Define dataset class for loading images and labels
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, labels_csv, transform=None):
        # Initialize dataset with image directory, labels CSV file, and optional transforms
        self.img_dir = img_dir
        self.labels = pd.read_csv(labels_csv)
        self.transform = transform

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.labels)

    def __getitem__(self, idx):
        # Get the image name and corresponding label for the given index
        img_name = self.labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file {img_path} not found.")
        image = Image.open(img_path).convert("RGB")  # Load image and convert to RGB
        label = 1 if self.labels.iloc[idx, 2] == "OK" else 0  # Convert label to binary (OK: 1, NOK: 0)

        if self.transform:
            # Apply transformations if provided
            image = self.transform(image)

        return {"image": image, "label": label}  # Return the transformed image and label

# Training loop for each worker
def train_loop_per_worker(config):
    # Load dataset and set up dataloaders
    data_dir = config["data_dir"]
    labels_path = config["labels_path"]
    batch_size = config["batch_size"]

    # Load data with Ray Data
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file {labels_path} not found.")
    data = rd.read_csv(labels_path)

    # Define image transformations (resize, convert to tensor, normalize)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 (required for ResNet)
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize using ImageNet stats
    ])

    # Convert Ray dataset to PyTorch dataset
    def preprocess_row(row):
        img_name = row["Part ID"]
        img_path = os.path.join(data_dir, img_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file {img_path} not found.")
        image = Image.open(img_path).convert("RGB")  # Load image and convert to RGB
        label = 1 if row["Class"] == "OK" else 0  # Convert label to binary (OK: 1, NOK: 0)
        if transform:
            image = transform(image)
        return {"image": image, "label": label}

    data_list = list(data.map(preprocess_row).iter_rows())
    dataset = [(x["image"], x["label"]) for x in data_list]
    train_size = int(0.8 * len(data_list))
    test_size = len(data_list) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Set up the model, loss function, and optimizer
    model = models.resnet18()  # Load a ResNet-18 model without pre-trained weights
    num_ftrs = model.fc.in_features  # Get the number of input features for the fully connected layer
    model.fc = nn.Linear(num_ftrs, 2)  # Replace the fully connected layer to match the number of classes (OK, NOK)

    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])  # SGD optimizer

    # Training loop
    scaler = torch.amp.GradScaler()  # Mixed precision scaler
    num_epochs = config["num_epochs"]
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for inputs, labels in train_loader:
            with torch.amp.autocast(device_type="cuda"):  # Mixed precision context
                outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            scaler.scale(loss).backward()  # Backward pass (compute gradients)
            scaler.step(optimizer)  # Update model parameters
            scaler.update()  # Update the scaler
            optimizer.zero_grad()  # Zero the gradient buffers

            running_loss += loss.item()  # Accumulate the loss

        avg_loss = running_loss / len(train_loader)  # Calculate average loss for the epoch
        session.report({"loss": avg_loss})  # Report metrics to Ray Tune

    # Evaluate the model on the test set
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    session.report({"test_accuracy": accuracy})  # Report test accuracy to Ray Tune

# Define the main training configuration
if __name__ == "__main__":
        # Start Ray runtime with Kubernetes configuration for distributed training
    ray.init(runtime_env={
        "working_dir": ".",
        "excludes": ["jammy-server-cloudimg-amd64.img"],
        "env_vars": {
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "NCCL_ASYNC_ERROR_HANDLING": "1"
        }
    })

    # Set training configuration
    config = {
        "data_dir": "/srv/nfs/kube-ray/visionline",  # Directory containing training images
        "labels_path": "/srv/nfs/kube-ray/labels.csv",  # Path to labels CSV file
        "batch_size": 32,  # Batch size for training
        "lr": 0.001,  # Learning rate for optimizer
        "num_epochs": 10  # Number of epochs to train the model
    }

    # Create a TorchTrainer to manage distributed training with Kubernetes
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=config,
        scaling_config=ScalingConfig(
            num_workers=1,  # Number of workers for distributed training
            use_gpu=True,  # Use GPU if available
            resources_per_worker={"CPU": 1, "GPU": 1}  # Resource allocation per worker
        ),
    )

    # Run the training
    result = trainer.fit()  # Start the training process
    print("Training complete with result:", result)  # Print the result of the training

    ray.shutdown()  # Shutdown Ray runtime