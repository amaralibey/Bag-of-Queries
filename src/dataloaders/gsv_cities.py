# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

"""
GSV-Cities dataset 
====================

This module implements a PyTorch Dataset class for GSV-Cities dataset from the paper:

GSV-Cities: Toward Appropriate Supervised Visual Place Recognition

Citation:
    @article{ali2022gsv,
        title={{GSV-Cities}: Toward appropriate supervised visual place recognition},
        author={Ali-bey, Amar and Chaib-draa, Brahim and Giguere, Philippe},
        journal={Neurocomputing},
        volume={513},
        pages={194--203},
        year={2022},
        publisher={Elsevier}
    }

URL: https://arxiv.org/abs/2210.10239
"""

import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T


# Transforms are passed to the dataset, if not, we will use this standard transform
default_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# NOTE: you can download the GSV-Cities dataset using the scripts we borrowed from 
# OpenVPRLab: https://github.com/amaralibey/OpenVPRLab
# the scripts are in the folder `scripts` in the root of the repository

class GSVCitiesDataset(Dataset):
    def __init__(
        self,
        dataset_path: str | Path = None,
        cities: str | list = "all",
        img_per_place: int = 4,
        transform=default_transform,
    ):
        """
        Args:
            base_path (Path): Base path for the dataset files.
            cities (list): List of city names to use in the dataset. Default is "all" or None which uses all cities.
            img_per_place (int): The number of images per place.
            transform (callable): Optional transform to apply on images.
        """
        super().__init__()
        
        self.base_path = self._validate_path(dataset_path)
        self.cities = self._validate_cities(cities)
        self.transform = transform
        self.img_per_place = img_per_place
        
        # Load and process dataframes
        self.dataframe = self._load_dataframes()
        self.places_ids = pd.unique(self.dataframe.index)
        self.total_nb_images = len(self.dataframe)
        
    def __getitem__(self, index):
        place_id = self.places_ids[index]        
        place = self.dataframe.loc[place_id]
        
        # randomly sample K images (rows) from this place
        place = place.sample(n=self.img_per_place) 
        
        # load and transform images
        imgs = []
        for _, row in place.iterrows():
            img_name = self._get_img_name(row)
            img_path = self.base_path / 'Images' / row['city_id'] / img_name
            # img = Image.open(img_path).convert('RGB')
            img = torchvision.io.decode_image(img_path, mode="RGB")
            
            
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)

        # NOTE: contrary to image classification where __getitem__ returns only one image 
        # in GSVCities, we return a place, which is a Tesor of K images (K=self.img_per_place)
        # this will return a Tensor of shape [K, channels, height, width]. This needs to be taken into account 
        # in the Dataloader (which will yield batches of shape [B, K, channels, height, width])
        return torch.stack(imgs), torch.tensor(place_id).repeat(self.img_per_place)

    @staticmethod
    def _get_img_name(row):
        """
            Given a row from the dataframe
            return the corresponding image name
        """
        city = row['city_id']
        # now remove the two digit we added to the id
        # they are superficially added to make ids different
        # for different cities
        pl_id = row.name % 10**5  #row.name is the index of the row, not to be confused with image name
        pl_id = str(pl_id).zfill(7)
        
        panoid = row['panoid']
        year = str(row['year']).zfill(4)
        month = str(row['month']).zfill(2)
        northdeg = str(row['northdeg']).zfill(3)
        lat, lon = str(row['lat']), str(row['lon'])
        img_name = f"{city}_{pl_id}_{year}_{month}_{northdeg}_{lat}_{lon}_{panoid}.jpg"
        return img_name
    
    def __len__(self):
        # return the total number of places (not images)
        return len(self.places_ids)
    
    def _validate_path(self, dataset_path):
        if dataset_path is None:
            dataset_path = Path(__file__).parent.parent.parent / "data/train/gsv-cities"
            
        path = Path(dataset_path)
        msg = "Make sure you downloaded the dataset using the provided script."
        if not path.exists():
            raise FileNotFoundError(f"GSV-Cities path {path} is not valid. {msg}")
        if not (path / 'Dataframes').exists():
            raise FileNotFoundError(f"Dataframes folder not found in gsv-cities {path}. {msg}")
        if not (path / 'Images').exists():
            raise FileNotFoundError(f"Images folder not found in gsv-cities {path}. {msg}")
        return path
        
    def _validate_cities(self, cities):
        if cities in ["all", None]:
            return [f.stem for f in self.base_path.glob("Dataframes/*.csv")]
        cities = [cities] if isinstance(cities, str) else cities
        for city in cities:
            if not (self.base_path / 'Dataframes' / f'{city}.csv').exists():
                raise FileNotFoundError(f"City {city} not found in dataset")
        return cities

    def _load_dataframes(self):
        """
            Return one dataframe containing
            all info about the images from all cities

            This requieres DataFrame files to be in a folder
            named Dataframes, containing one DataFrame
            for each city in self.cities
        """
        dataframes = []
        for i, city in enumerate(self.cities):
            df = pd.read_csv(self.base_path / 'Dataframes' / f'{city}.csv')
            df['place_id'] += i * 10**5 # to avoid place_id conflicts between cities
            df = df.sample(frac=1) # shuffle within city
            dataframes.append(df)
        
        df = pd.concat(dataframes)
        # keep only places depicted by at least `self.img_per_place` images
        df = df[df.groupby('place_id')['place_id'].transform('size') >= self.img_per_place]
        return df.set_index('place_id')
    
    def _refresh_dataframes(self):
        self.dataframe = self._load_dataframes()
        self.places_ids = pd.unique(self.dataframe.index)
        self.total_nb_images = len(self.dataframe)