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

from src.dataloaders import GSVCitiesDataset
from src.dataloaders import PittsburghDataset
from src.dataloaders import MapillarySLSDataset

class VPRDataModule(L.LightningDataModule):
    def __init__(
        self,
        gsv_cities_path: str = None,
        cities: str | list = "all",
        img_per_place: int = 4,
        val_sets: dict = {"msls-val": None, "pitts30k-val": None},
        train_img_size=(224, 224),
        val_img_size=(224, 224),
        batch_size: int = 100,
        num_workers: int = 8,
        shuffle: bool = False,
        mean_std: dict = {"mean":[0.485, 0.456, 0.406], "std":[0.229, 0.224, 0.225]},
    ):
        super().__init__()
        self.gsv_cities_path = gsv_cities_path
        self.cities = cities
        self.img_per_place = img_per_place
        self.val_sets = val_sets
        self.train_img_size = train_img_size
        self.val_img_size = val_img_size
        self.mean_std = mean_std
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        
        self.train_transform = T.Compose([
            T.Resize(train_img_size, interpolation=3),
            T.RandAugment(num_ops=3, magnitude=15, interpolation=2),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(**mean_std),
        ])
        
        self.val_transform = T.Compose([
            T.Resize(val_img_size, interpolation=3),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(**mean_std),
        ])

    def setup(self, stage=None):
        if stage in ["fit", "reload", None]:
            self.train_dataset = GSVCitiesDataset(
                dataset_path=self.gsv_cities_path,
                cities=self.cities,
                img_per_place=self.img_per_place,
                transform=self.train_transform
            )
        
        if stage == "fit" or stage is None:
            self.val_datasets = []
            if "msls-val" in self.val_sets:
                val_ds = MapillarySLSDataset(
                    dataset_path=self.val_sets["msls-val"],
                    transform=self.val_transform
                )
                self.val_datasets.append(val_ds)
            if "pitts30k-val" in self.val_sets:
                val_ds = PittsburghDataset(
                    dataset_path=self.val_sets["pitts30k-val"],
                    transform=self.val_transform
                )
                self.val_datasets.append(val_ds)
        
    def train_dataloader(self):
        self.setup(stage="reload") # reload the train dataset
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=self.shuffle
        )

    def val_dataloader(self):
        return [
            DataLoader(
                val_ds,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
            ) for val_ds in self.val_datasets
        ]