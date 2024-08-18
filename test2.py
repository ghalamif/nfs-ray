import torch
from torch import nn
import ray
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

def train_func():
    model = nn.Sequential(nn.Linear(30, 1), nn.Sigmoid())
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # Datasets can be accessed in your train_func via ``get_dataset_shard``.
    train_data_shard = train.get_dataset_shard("train")

    for epoch_idx in range(2):
        for batch in train_data_shard.iter_torch_batches(batch_size=128, dtypes=torch.float32):
            features = torch.stack([batch[col_name] for col_name in batch.keys() if col_name != "target"], axis=1)
            predictions = model(features)
            train_loss = loss_fn(predictions, batch["target"].unsqueeze(1))
            train_loss.backward()
            optimizer.step()


train_dataset = ray.data.read_csv("s3://anonymous@air-example-data/breast_cancer.csv")

trainer = TorchTrainer(
    train_func,
    datasets={"train": train_dataset},
    scaling_config=ScalingConfig(num_workers=2)
)
trainer.fit()