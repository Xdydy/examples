import torch.distributed.rpc as rpc

rpc.init_rpc('driver',rank=0)