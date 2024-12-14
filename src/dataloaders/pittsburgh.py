# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

from typing import Optional, Callable, Tuple, Any
import pathlib
import numpy as np
from pathlib import Path
import torchvision
from torch.utils.data import Dataset
from PIL import Image


# NOTE: for pitts30k-test and pitts250k-test 
# you need to download them from  the author's website
# https://www.di.ens.fr/willow/research/netvlad/
# 
# For faster loading I hardcoded the image names and ground truth for pitts30k-val (already comes with OpenVPRLab)

REQUIRED_FILES = {
    "pitts30k-val":     ["pitts30k_val_dbImages.npy", "pitts30k_val_qImages.npy", "pitts30k_val_gt_25m.npy"],
}

class PittsburghDataset(Dataset):
    """
    Args:
        dataset_path (str): Directory containing the dataset. If None, the path `data/val/pitts30k-val` will be used.
        input_transform (callable, optional): Optional transform to be applied on each image.
    
    Reference:
        @inproceedings{torii2013visual,
            title={Visual place recognition with repetitive structures},
            author={Torii, Akihiko and Sivic, Josef and Pajdla, Tomas and Okutomi, Masatoshi},
            booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
            pages={883--890},
            year={2013}
        }
    """

    def __init__(
        self,
        dataset_path: Optional [str] = None,
        transform: Optional[Callable] = None,
    ):
        
        self.transform = transform
        dataset_path = self._validate_path(dataset_path)
        self.dataset_path = dataset_path
        self.dataset_name = dataset_path.name
        
        # load image names and ground truth data
        self.dbImages = np.load(dataset_path / REQUIRED_FILES[self.dataset_name][0])
        self.qImages = np.load(dataset_path / REQUIRED_FILES[self.dataset_name][1])
        self.ground_truth = np.load(dataset_path / REQUIRED_FILES[self.dataset_name][2], allow_pickle=True)

        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages))
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

        # combine reference and query images
        self.image_paths = np.concatenate((self.dbImages, self.qImages))
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)
        
    def __getitem__(self, index: int) -> Tuple[Any, int]:
        img_path = self.image_paths[index]
        # img = Image.open(self.dataset_path / img_path)
        img = torchvision.io.decode_image(self.dataset_path / img_path, mode="RGB")

        if self.transform:
            img = self.transform(img)

        return img, index

    def __len__(self) -> int:
        return len(self.image_paths)
    
    def _validate_path(self, dataset_path):
        
        if dataset_path is None:
            dataset_path = Path(__file__).parent.parent.parent / "data" / "val" / "msls-val"
        
        path = Path(dataset_path)

        msg = "Make sure you downloaded the dataset with the provided script."
        if not path.is_dir():
            raise FileNotFoundError(f"The directory {dataset_path} does not exist. {msg}")
        
        # make sure required metadata files are in the directory        
        if not all((path / file).is_file() for file in REQUIRED_FILES[path.name]):
            raise FileNotFoundError(f"Please make sure all requiered metadata for {dataset_path} are in the directory. i.e. {REQUIRED_FILES[self.dataset_name]}")
        
        return path
