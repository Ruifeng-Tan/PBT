import abc
import torch
import shutil


class BaseModel(abc.ABC):
    """Scikit-learn-like interface for models."""
    def __init__(self, workspace: str = None):
        self.workspace = workspace

    @abc.abstractmethod
    def dump_checkpoint(self, path: str):
        """Dump checkpoint to disk."""

    @abc.abstractmethod
    def load_checkpoint(self, path: str):
        """Load checkpoint from disk."""

    def to(self, device: str):
        """Move the model to the device."""
        return self

    def link_latest_checkpoint(self, filename: str):
        to_dump = self.workspace / 'latest.ckpt'
        shutil.copyfile(filename, to_dump)
