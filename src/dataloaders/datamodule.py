# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import torch
import lightning as L
from torch.utils.data.dataloader import Dataset, DataLoader
from torchvision.transforms import v2  as T


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        train_set_name: str = "gsv-cities",
        val_set_names: list = ["msls-val", "pitts30k-val"],
        train_img_size=(224, 224),
        val_img_size=(224, 224),
        batch_size: int = 100,
        num_workers: int = 8,
        pin_memory: bool = True,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        super().__init__()
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass