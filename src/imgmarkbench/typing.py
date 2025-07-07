import torch
from enum import Enum


class AggregatorType(str, Enum):
    csv = "csv"
    clickhouse = "clickhouse"


class ExecutorType(str, Enum):
    thread = "thread"
    process = "process"

# ToDo: may be jaxtyping?
TorchImg = torch.Tensor
'''
 Image is represented as float32 torch tensor of shape (C x H x W) in the range [0.0, 1.0], channels RGB 
'''
