from .dataset import FITSDataset
from .bin_dataset import BinaryFITSDataset

from torch.utils.data import DataLoader

def get_data_loader(dataset, batch_size, n_workers, shuffle=True):
    return DataLoader(dataset,
                      batch_size=int(batch_size),
                      shuffle=shuffle,
                      num_workers=n_workers,
    )


__all__ = ["FITSDataset", "BinaryFITSDataset", "get_data_loader"]
