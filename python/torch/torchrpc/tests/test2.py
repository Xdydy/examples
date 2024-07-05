import torch.distributed.rpc as rpc
import time
import os
os.environ['MASTER_ADDR'] = "localhost"
os.environ['MASTER_PORT'] = "8000"
rpc.init_rpc('worker1',rank=1)

time.sleep(3600)