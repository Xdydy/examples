import pandas as pd
import numpy as np

def load_model():
    def model(batch: pd.DataFrame) -> pd.DataFrame:
        model.payload = np.zeros(100_000_000)
        return pd.DataFrame({'score': batch['passenger_count'] % 2 == 0})
    return model

import pyarrow.parquet as pq
import ray

@ray.remote
def make_prediction(model, shard_path):
    df = pq.read_table(shard_path).to_pandas()
    result = model(df)
    return len(result)


# 12 files, one for each remote task.
input_files = [
        f"s3://anonymous@air-example-data/ursa-labs-taxi-data/downsampled_2009_full_year_data.parquet"
        f"/fe41422b01c04169af2a65a83b753e0f_{i:06d}.parquet"
        for i in range(12)
]

# ray.put() the model just once to local object store, and then pass the
# reference to the remote tasks.
model = load_model()
model_ref = ray.put(model)

result_refs = []
print(type(make_prediction))
# Launch all prediction tasks.
for file in input_files:
    # Launch a prediction task by passing model reference and shard file to it.
    # NOTE: it would be highly inefficient if you are passing the model itself
    # like make_prediction.remote(model, file), which in order to pass the model
    # to remote node will ray.put(model) for each task, potentially overwhelming
    # the local object store and causing out-of-disk error.
    result_refs.append(make_prediction.remote(model_ref, file))

results = ray.get(result_refs)

# Let's check prediction output size.
for r in results:
    print("Prediction output size:", r)