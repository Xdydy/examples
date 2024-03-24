# Note that this example class is doing the same thing as the basic DataConfig
# implementation included with Ray Train.
from typing import Optional, Dict, List

import ray
from ray import train
from ray.train.torch import TorchTrainer
from ray.train import DataConfig, ScalingConfig
from ray.data import Dataset, DataIterator, NodeIdStr
from ray.actor import ActorHandle

ds = ray.data.read_text(
    "s3://anonymous@ray-example-data/sms_spam_collection_subset.txt"
)

def train_loop_per_worker():
    # Get an iterator to the dataset we passed in below.
    it = train.get_dataset_shard("train")
    for _ in range(2):
        for batch in it.iter_batches(batch_size=128):
            print("Do some training on batch", batch)


class MyCustomDataConfig(DataConfig):
    def configure(
        self,
        datasets: Dict[str, Dataset],
        world_size: int,
        worker_handles: Optional[List[ActorHandle]],
        worker_node_ids: Optional[List[NodeIdStr]],
        **kwargs,
    ) -> List[Dict[str, DataIterator]]:
        assert len(datasets) == 1, "This example only handles the simple case"

        # Configure Ray Data for ingest.
        ctx = ray.data.DataContext.get_current()
        ctx.execution_options = DataConfig.default_ingest_options()

        # Split the stream into shards.
        iterator_shards = datasets["train"].streaming_split(
            world_size, equal=True, locality_hints=worker_node_ids
        )

        # Return the assigned iterators for each worker.
        return [{"train": it} for it in iterator_shards]


my_trainer = TorchTrainer(
    train_loop_per_worker,
    scaling_config=ScalingConfig(num_workers=2),
    datasets={"train": ds},
    dataset_config=MyCustomDataConfig(),
)
my_trainer.fit()