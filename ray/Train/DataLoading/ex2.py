import ray
import ray.data
import ray.train
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

ds = ray.data.read_text("s3://anonymous@ray-example-data/sms_spam_collection_subset.txt")
train_ds, val_ds = ds.train_test_split(0.3)

def train_loop_per_worker():
    train_ds = train.get_dataset_shard("train")
    for _ in range(2):
        for batch in train_ds.iter_batches(batch_size=128):
            print("Do some training on batch", batch)

    val_ds = train.get_dataset_shard("val")
    for _ in range(2):
        for batch in val_ds.iter_batches(batch_size=128):
            print("Do some validation on batch", batch)

my_trainer = TorchTrainer(
    train_loop_per_worker,
    scaling_config=ScalingConfig(
        num_workers=2,
    ),
    datasets={"train": train_ds, "val": val_ds},
    dataset_config=ray.train.DataConfig(
        datasets_to_split=['train']
    )
)
my_trainer.fit()