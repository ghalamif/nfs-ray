import os
import torch
from torch.nn import CrossEntropyLoss, Linear
from torch.optim import Adam
from typing import Any, Dict
from torch.cuda.amp import GradScaler
#import numpy as np
from torchvision import transforms
#from PIL import Image
import pandas as pd
from ray import train, tune
from ray.tune import Tuner, TuneConfig
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from ray.data import DataContext
import ray.train.torch
from torchvision.models import resnet18
from ray.train import FailureConfig



#import logging
import tempfile
#logging.basicConfig(level=logging.DEBUG)




context = DataContext.get_current()
context.verbose_stats_logs = True
ray.data.DataContext.get_current().DEFAULT_ENABLE_PROGRESS_BAR_NAME_TRUNCATION = False

def parse_filename(row: Dict[str, Any]) -> Dict[str, Any]:
    row["filename"] = os.path.basename(row["path"])
    return row

def transform_image(row: Dict[str, Any]) -> Dict[str, Any]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        #transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.Resize((128, 128)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    row["image"] = transform(row["image"])
    return row

# Step 3: Load labels.csv using pandas and convert it to a dictionary
labels_df = pd.read_csv("/srv/nfs/kube-ray/labels.csv")
#print(labels_df.columns)  # Print column names to verify them

# Use 'Class' as the label column, as found in the CSV file
labels_dict = pd.Series(labels_df['Class'].values, index=labels_df['Part ID']).to_dict()

# Step 4: Define a function to add labels to the dataset rows based on filename

def add_label(row: Dict[str, Any]) -> Dict[str, Any]:
    part_id = row["filename"]
    row["label"] = 1 if labels_dict.get(part_id, None) == "OK" else 0  # Add label if exists, else set to None
    return row

def convert_to_torch(row: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    # Only convert the image tensor, ensure it's of type float32
    row["tensor"] = torch.tensor(row["image"], dtype=torch.float32)
    row["label"] = torch.tensor(row["label"], dtype=torch.long)  # Convert label to tensor as well
    return row

def filter_numerical_columns(row: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    # Only keep numerical columns that are needed for training
    return {key: row[key] for key in ["tensor", "label"]}

processed_image_ds = ray.data.read_images("/srv/nfs/kube-ray/visionline", mode="RGB", include_paths=True)\
    .map(transform_image)\
    .map(parse_filename)\
    .map(add_label)\
    .map(convert_to_torch)\
    .map(filter_numerical_columns)

def train_func(config: Dict):
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size_per_worker"]
  
    model = resnet18(num_classes=2)
    num_ftrs = model.fc.in_features  # Get the number of input features for the fully connected layer
    model.fc = Linear(num_ftrs, 2)  # Replace the fully connected layer to match the number of classes (OK, NOK)

    # Prepare and wrap your model with DistributedDataParallel
    model = ray.train.torch.prepare_model(model)

    # Set up loss function and optimizer
    criterion = CrossEntropyLoss()  # Cross-entropy loss for classification
    optimizer = Adam(model.parameters(), lr=lr)
    #scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())



    '''
    # Get training and test dataset shards
    train_data_shard = train.get_dataset_shard("train")
    test_data_shard = train.get_dataset_shard("test")

    # Create dataloaders for train and test datasets
    
    train_dataloader = train_data_shard.iter_torch_batches(
        batch_size=batch_size, dtypes=torch.float32
    )
    '''
    
    # Get training and test dataset shards
    train_data_shard = train.get_dataset_shard("train")
    test_data_shard = train.get_dataset_shard("test")

    #train_data_shard, test_data_shard = get_dataloaders(batch_size=batch_size)
    
    # Create dataloaders for train and test datasets
    train_dataloader = train_data_shard.iter_torch_batches(
        batch_size=batch_size, dtypes=torch.float32
    )
    test_dataloader = test_data_shard.iter_torch_batches(
        batch_size=batch_size, dtypes=torch.float32
    )

    
    #train_dataloader = ray.train.torch.prepare_data_loader(train_data_shard)
    #test_dataloader = ray.train.torch.prepare_data_loader(test_data_shard)

    # Training loop
    for epoch in range(epochs):
        model.train() # Set model to training mode
        running_loss = 0.0
        batch_count = 0
        batch_test_count = 0
        #if ray.train.get_context().get_world_size() > 1:
            #train_dataloader.sampler.set_epoch(epoch)

        # Training Step
        for batch in train_dataloader:
            # Ensure only numerical columns (tensor and label) are included in the batch
            images, labels = batch["tensor"].float(), batch["label"].long()
            num_images = images.shape[0]

            # Zero the parameter gradients
            optimizer.zero_grad()

            if torch.cuda.is_available():
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)  # Forward pass
                    loss = criterion(outputs, labels)  # Compute the loss
            else:
                outputs = model(images)  # Forward pass
                loss = criterion(outputs, labels)  # Compute the loss


            # Backward pass and optimize
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()  # Accumulate the loss
            batch_count += 1

        avg_loss = running_loss / batch_count if batch_count > 0 else float('inf')
        metrics = {"loss": avg_loss, "epoch": epoch}
        
        
        # Evaluation Step
        model.eval()  # Set model to evaluation mode
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in test_dataloader:
                images, labels = batch["tensor"].float(), batch["label"].long()
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                batch_test_count += 1

        test_loss = total_loss / batch_test_count if batch_test_count > 0 else float('inf')
        accuracy = correct / total

        # Update Metrics
        metrics = {
            "train_loss": avg_loss,
            "epoch": epoch,
            "test_loss": test_loss,
            "test_accuracy": accuracy,
            "lr": lr,
            "batch_size": batch_size,
            "num_workers": ray.train.get_context().get_world_size(),
            "use_gpu": torch.cuda.is_available(),
            "image number": num_images
        }

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
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

train_dataset, test_dataset = processed_image_ds.train_test_split(test_size=0.25, shuffle=True)

# [4] Define the training configuration.
total_batch_size = 16
num_workers = 2
use_gpu = True

#train_config = {
    #"lr": 1e-5,
    #"lr": tune.loguniform(1e-5, 1e-1),
    #"epochs": 12,
    #"batch_size_per_worker": total_batch_size // num_workers,
#}

train_config = {
    "lr": 0.0005,  # Learning rate range for tuning
    "epochs": 20,                       # Fixed number of epochs
    "batch_size_per_worker": total_batch_size // num_workers,        # Batch size for each worker
}

# [5] Launch distributed training job.
trainer = TorchTrainer(
    train_func,
    train_loop_config=train_config,
    datasets={"train": train_dataset, "test": test_dataset},
    scaling_config=ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu
    ),
    run_config=ray.train.RunConfig(
        storage_path="/srv/nfs/kube-ray",
        name="storage",
        sync_config=ray.train.SyncConfig(sync_artifacts=True),
    ),
)



results = trainer.fit()
ray.shutdown()

# Load the trained model
#with result.checkpoint.as_directory() as checkpoint_dir:
    #model_state_dict = torch.load(os.path.join(checkpoint_dir, "model.pt"))
    #model = resnet18(num_classes=2)
    #model.load_state_dict(model_state_dict)