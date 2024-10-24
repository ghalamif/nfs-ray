import os
import torch
from torch.nn import CrossEntropyLoss, Linear
from torch.optim import Adam
from typing import Any, Dict
from torch.cuda.amp import GradScaler
import numpy as np
from torchvision import transforms
from PIL import Image
import pandas as pd
import ray
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from ray.data import DataContext
import ray.train.torch
from torchvision.models import resnet18
from ray.train import FailureConfig

import logging
import tempfile
logging.basicConfig(level=logging.DEBUG)

context = DataContext.get_current()
context.verbose_stats_logs = True

def parse_filename(row: dict[str, Any]) -> dict[str, Any]:
    row["filename"] = os.path.basename(row["path"])
    return row

def transform_image(row: Dict[str, Any]) -> Dict[str, Any]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    row["image"] = transform(row["image"])
    return row

# Step 3: Load labels.csv using pandas and convert it to a dictionary
labels_df = pd.read_csv("/srv/nfs/kube-ray/labels.csv")
print(labels_df.columns)  # Print column names to verify them

# Use 'Class' as the label column, as found in the CSV file
labels_dict = pd.Series(labels_df['Class'].values, index=labels_df['Part ID']).to_dict()

# Step 4: Define a function to add labels to the dataset rows based on filenameProvider
def add_label(row: Dict[str, Any]) -> Dict[str, Any]:
    part_id = row["filename"]
    row["label"] = 1 if labels_dict.get(part_id, None)=="OK" else 0  # Add label if exists, else set to None
    return row

def convert_to_torch(row: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    row["tensor"] = torch.as_tensor(row["image"])
    return row

processed_image_ds = ray.data.read_images("/srv/nfs/kube-ray/visionline", mode="RGB", include_paths=True)\
.map(transform_image)\
.map(parse_filename)\
.map(add_label)\
.map(convert_to_torch)

print(processed_image_ds.take(1))
print(processed_image_ds.schema())

def train_func(config: Dict):
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size_per_worker"]
    model = resnet18(num_classes=2)
    num_ftrs = model.fc.in_features  # Get the number of input features for the fully connected layer
    model.fc = Linear(num_ftrs, 2)  # Replace the fully connected layer to match the number of classes (OK, NOK)


    # [1] Prepare model.
    # Prepare and wrap your model with DistributedDataParallel
    # Move the model to the correct GPU/CPU device
    model = ray.train.torch.prepare_model(model)

    # Set up loss function and optimizer
    criterion = CrossEntropyLoss()  # Cross-entropy loss for classification
    #optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])  # SGD optimizer
    optimizer = Adam(model.parameters(), lr=lr)

    #optimizer = Adam(model.parameters(), lr)
    scaler = GradScaler()

    # Datasets can be accessed in your train_func via `get_dataset_shard.
    train_data_shard = train.get_dataset_shard("train")
    # iter_torch_batches returns an iterable object that
    # yield tensor batches. Ray Data automatically moves the Tensor batches
    # to GPU if you enable GPU training.
    train_dataloader = train_data_shard.iter_torch_batches(
        batch_size=batch_size, dtypes=torch.float32
    )

    # Training
    for epoch in range (epochs):
        running_loss = 0.0
        if ray.train.get_context().get_world_size() > 1:
            train_dataloader.sampler.set_epoch(epoch)

        for batch in train_dataloader:
            images, labels = batch["tensor"], batch["label"]
            
            # Zero the parameter gradients
            optimizer. zero_grad()

            # Forward pass, backward pass, and optimize
            outputs = model(images) # Forward pass
            loss = criterion(outputs, labels) # Compute the loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()  # Accumulate the loss

        #avg_loss = running_loss / len(train_dataloader)  # Calculate average loss for the epoch
        
        # [3] Report metrics and checkpoint.
        #metrics = {"loss": loss.item(),"Avg-loss": avg_loss , "epoch": epoch}
        metrics = {"loss": loss.item(), "epoch": epoch}
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            if ray.train.get_context().get_world_rank() == 1:
                state_dict = model.state_dict()
            else:
                state_dict = model.module.state_dict()
                
            torch.save(
                state_dict,
                os.path.join(temp_checkpoint_dir, "model.pt")
            )
            ray.train.report(
                metrics=metrics,
                checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
            )
        if ray.train.get_context().get_world_rank() == 0:
            print(metrics)



train_dataset, test_dataset = processed_image_ds.train_test_split(test_size=0.25, shuffle=True) # Materialize and split the dataset into train and test subsets.This may be a very expensive operation with a large dataset.

# [4] Define the training configuration.
total_batch_size = 8
num_workers = 1
use_gpu=True

train_config = {
    "lr": 1e-3,
    "epochs": 10,
    "batch_size_per_worker": total_batch_size // num_workers,
}
# [5] Launch distributed training job.
trainer = TorchTrainer(
    train_func,
    train_loop_config=train_config,
    datasets={"train": train_dataset, "test": test_dataset},
    scaling_config=ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu
    ),  # Close the ScalingConfig parentheses here
    # [5a] If running in a multi-node cluster, this is where you
    # should configure the run's persistent storage that is accessible
    # across all worker nodes.
    run_config=ray.train.RunConfig(storage_path="/srv/nfs/kube-ray",
                                    name="storage",
                                    sync_config=ray.train.SyncConfig(sync_artifacts=True),
                                    failure_config=FailureConfig(max_failures=2),
                                    ),
)

result = trainer.fit()
print("Observed metrics:", result.metrics)
df = result.metrics_dataframe #retrieve a pandas DataFrame of all reported metrics.
print("Minimum loss", min(df["loss"]))

# Print available checkpoints
for checkpoint, metrics in result.best_checkpoints:
    print("Loss", metrics["loss"], "checkpoint", checkpoint)

if result.error:
    assert isinstance(result.error, Exception)

    print("Got exception:", result.error)




#env_vars = {
    #"TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    #"NCCL_DEBUG":"INFO",
#}

#ray.init()
