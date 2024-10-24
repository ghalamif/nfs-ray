import os
import pandas as pd
from PIL import Image
from filelock import FileLock
from typing import Dict

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.amp import GradScaler, autocast
import torchvision

import ray
from ray import train
from ray.train.torch import TorchTrainer, prepare_model, prepare_data_loader
from ray.train import report
from ray.train import FailureConfig

import logging
logging.basicConfig(level=logging.DEBUG)

# Load data function with file locking for safe downloads
def load_data(csv_file, root_dir, transform):
    with FileLock(os.path.expanduser("~/data.lock")):
        labels = pd.read_csv(csv_file)
        data = []
        for idx in range(len(labels)):
            img_name = os.path.join(root_dir, labels.iloc[idx, 0])
            image = Image.open(img_name).convert("RGB")
            label = 1 if labels.iloc[idx, 2] == "OK" else 0
            if transform:
                image = transform(image)
            data.append((image, label))
    return data


def get_dataloaders(batch_size):
    transform = torchvision.transforms.Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data_dir = '/srv/nfs/kube-ray/visionline/'
    labels_csv = '/srv/nfs/kube-ray/labels.csv'

    train_data = load_data(csv_file=labels_csv, root_dir=data_dir, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    return train_loader

# Training function with improved progress and metric reporting

def train_func_per_worker(config: Dict):
    device ='cpu' #"cuda" if torch.cuda.is_available() else "cpu"
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size_per_worker"]

    # Prepare Dataloader for distributed training
    #train_data_shard = train.get_dataset_shard("train")
    train_dataloader = get_dataloaders(batch_size=batch_size)
    train_loader = ray.train.torch.prepare_data_loader(train_dataloader)

    model = resnet18(num_classes=2)

    # Prepare and wrap your model with DistributedDataParallel
    # Move the model to the correct GPU/CPU device
    model = ray.train.torch.prepare_model(model)

    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr)
    scaler = GradScaler(device=device)

    for epoch in range(epochs):
        # Setting the epoch for distributed training
        if ray.train.get_context().get_world_size() > 1:
            train_loader.sampler.set_epoch(epoch)

        model.train()

        total_loss = 0
        total_samples = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast(device_type=device):
                outputs = model(images)
                loss = loss_function(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            # Clear unused memory
            if device == "cuda":
                torch.cuda.empty_cache()

        avg_loss = total_loss / total_samples
        metrics = {"loss": loss.item(), "avg_loss": avg_loss, "epoch": epoch}

        # Report metrics to Ray Train
        ray.train.report(metrics=metrics)

        # Print metrics in the console for the worker with rank 0
        if ray.train.get_context().get_world_size() == 0:
            print(metrics)

def train(num_workers, use_gpu):

    total_batch_size = 8
    train_config = {
        "lr": 1e-3,
        "epochs": 10,
        "batch_size_per_worker": total_batch_size // num_workers,
    }

    # Scaling config for training
    scaling_config = ray.train.ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu,)
    
    # Launch distributed training job
    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
    )

    result = trainer.fit()
    print(f"Training result: {result}")


if __name__ == "__main__":

    env_vars = {
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        #"NCCL_SOCKET_IFNAME": "ens5"
        "NCCL_DEBUG":"INFO",
    }

    #runtime_env = ray.runtime_env.RuntimeEnv(
        #pip=["pandas","torch","torchvision","ray[default]"],
        #env_vars={"TORCH_NCCL_ASYNC_ERROR_HANDLING": "1"},
        #working_dir="/srv/nfs/kube-ray"
    #)

    #ray.init(
    #runtime_env=runtime_env,
    #)

    #ray.init(address="ray://127.0.0.1:8083")
    #ray.init(address='auto' ,runtime_env=env_vars)
    ray.init(runtime_env=env_vars)
    print("Ray initialized with", ray.get_runtime_context())
    print("Ray version:", ray.__version__)
    print("Torch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name)
    print("CUDA version:", torch.version.cuda)
    print("CUDNN version:", torch.backends.cudnn.version())
    print("NCCL version:", torch.cuda.nccl.version())
    print("NCCL socket interface:", os.environ.get("NCCL_SOCKET_IFNAME"))
    print("NCCL async error handling:", os.environ.get("TORCH_NCCL_ASYNC_ERROR_HANDLING"))
    print("Working directory:", os.getcwd())
    print(ray.cluster_resources())
    print("PyTorch sees GPUs:", torch.cuda.device_count())
    print("Ray sees GPUs:", ray.cluster_resources().get('GPU', 0))

   

    train(num_workers=1, use_gpu=True)