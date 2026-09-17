import abc
import torch
import random
import numpy as np
import torch.nn as nn
from .base import BaseModel


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class NNModel(BaseModel, nn.Module, abc.ABC):
    def __init__(self,
                 batch_size: int = 32,
                 epochs: int = 10000,
                 workspace: str = None,
                 evaluate_freq: int = 500,
                 checkpoint_freq: int = 1000,
                 train_batch_size: int = None,
                 test_batch_size: int = None,
                 seed: int = None,
                 lr: float = 1e-3):
        nn.Module.__init__(self)
        BaseModel.__init__(self, workspace)
        self.train_epochs = epochs
        self.evaluate_freq = evaluate_freq
        if checkpoint_freq is None or checkpoint_freq == 'None':
            self.checkpoint_freq = None
        else:
            self.checkpoint_freq = min(checkpoint_freq, self.train_epochs)
        self.train_batch_size = train_batch_size or batch_size
        self.test_batch_size = test_batch_size or batch_size
        self.lr = lr
        self.seed = seed

    def to(self, device: str):
        return nn.Module.to(self, device)

    def dump_checkpoint(self, path: str):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: str, device: str="cuda"):
        self.load_state_dict(torch.load(path, map_location=torch.device(device)))

def reset_parameters(model):
    @torch.no_grad()
    def weight_reset(m):
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    model.apply(weight_reset)
