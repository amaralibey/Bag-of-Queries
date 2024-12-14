# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import argparse

import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from src.backbones import DinoV2, ResNet
from src.boq import BoQ
from src.model import BoQModel
from src.dataloaders import GSVCitiesDataset, MapillarySLSDataset, PittsburghDataset


class HyperParams:
    # backbone
    backbone_name: str = "dinov2_vitb14"
    unfreeze_n_blocks: int = 2
    
    # BoQ
    channel_proj: int = 512
    num_queries: int = 64
    num_layers: int = 2
    output_dim: int = 8192
    
    # training
    batch_size: int = 128
    max_epochs: int = 40
    warmup_epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-6
    lr_mul: float = 0.1
    milestones: list = [10, 20]
    num_workers: int = 8
    
    # misc
    compile: bool = False
    seed: int = 2024

def train(hparams, dev_mode=False):
    seed_everything(hparams.seed, workers=True)
    
    # Instantiate the backbone and define the image size for training and validation
    if "dinov2" in hparams.backbone_name:
        backbone = DinoV2(backbone_name=hparams.backbone_name, unfreeze_n_blocks=hparams.unfreeze_n_blocks)
        hparams.backbone_name = backbone.backbone_name # in case the user passed dinov2 without the version
        train_image_size = (224, 224)
        val_image_size = (322, 322)
    elif "resnet" in hparams.backbone_name:
        backbone = ResNet(backbone_name=hparams.backbone_name, unfreeze_n_blocks=hparams.unfreeze_n_blocks, crop_last_block=True)
        train_image_size = (320, 320)
        val_image_size = (384, 384)
    else:
        raise ValueError(f"backbone note recognized: {hparams.backbone_name}") 
    
    
    # Instantiate BoQ aggregator
    aggregator = BoQ(
        in_channels=backbone.out_channels,
        proj_channels=hparams.channel_proj,
        num_queries=hparams.num_queries,
        num_layers=hparams.num_layers,
        row_dim=hparams.output_dim//hparams.channel_proj,
    )
    
    # Define the entire Lightning model for training and validation
    model = BoQModel(
        backbone,
        aggregator,
        lr=hparams.lr,
        lr_mul=hparams.lr_mul,
        weight_decay=hparams.weight_decay,
        warmup_epochs=hparams.warmup_epochs,
        milestones=hparams.milestones,
    )
    
    if hparams.compile:
        model = torch.compile(model)
    
    # Define the train and validation transforms
    train_transform = T.Compose([
        T.Resize(train_image_size, interpolation=3),
        T.RandAugment(num_ops=2, magnitude=15, interpolation=2),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = T.Compose([
        T.Resize(val_image_size, interpolation=3),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Let's define the training and validation datasets
    
    # If you want to train on a specific city, you can pass a list of cities (check the gsv-cities folder for the city names)
    # cities = [
    #     "paris",
    #     "london",
    #     # ...
    # ]
    cities = "all"
    train_dataset = GSVCitiesDataset(
        dataset_path="../OpenVPRLab/data/train/gsv-cities",     # path to the gsv-cities dataset (check ./scripts to download) 
        cities=cities, 
        img_per_place=4, 
        transform=train_transform)
    
    msls_val = MapillarySLSDataset(
        dataset_path="../OpenVPRLab/data/val/msls-val",         # path to the msls-val dataset (check ./scripts to download)
        transform=val_transform)
    
    pitts30k_val = PittsburghDataset(
        dataset_path="../OpenVPRLab/data/val/pitts30k-val",     # path to the pitts30k-val dataset (check ./scripts to download)
        transform=val_transform)
    
    # Define the dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        pin_memory=True,
        shuffle=False,
    )
    
    msls_val_dataloader = DataLoader(
        msls_val,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        pin_memory=True,
    )
    
    pitts30k_val_dataloader = DataLoader(
        pitts30k_val,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        pin_memory=True,
    )

    # we use Tensorboard for logging (integrated with PyTorch Lightning)
    tensorboard_logger = TensorBoardLogger(
        save_dir=f"./logs",
        name=f"{hparams.backbone_name}",
        default_hp_metric=False
    )
    
    # Define the checkpointing callback
    checkpointing = ModelCheckpoint(
        monitor="msls-val/R@1",
        filename="epoch[{epoch:02d}]_R@1[{msls-val/R@1:.4f}]_R@5[{msls-val/R@5:.4f}]",
        auto_insert_metric_name=False,
        save_weights_only=False,
        save_top_k=3,
        mode="max",
    )
    
    # Define the trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=[0],
        logger=tensorboard_logger,      # comment this line if you don't want to use tensorboard logger
        precision="16-mixed",
        callbacks=[
            checkpointing,              # this callback saves the best model based on the metric we monitor (recall@5)
            RichProgressBar()           # comment this line if you want classic progress bar
        ],
        max_epochs=hparams.max_epochs,
        reload_dataloaders_every_n_epochs=2,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
        enable_model_summary=True,
        fast_dev_run=dev_mode,
    )
    
    # Train the model
    trainer.fit(
        model=model, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=[msls_val_dataloader, pitts30k_val_dataloader]
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train parameters")

    parser.add_argument("--dev",        action="store_true", help="Enable fast dev run (one train and validation iteration).")
    parser.add_argument('--compile',    action='store_true', help='Compile the model using torch.compile()')
    
    parser.add_argument("--seed",   type=int,   help="Random seed for reproducibility.")
    
    parser.add_argument("--bs",     type=int,   help="Batch size.")
    parser.add_argument("--lr",     type=float, help="Learning Rate.")
    parser.add_argument("--wd",     type=float, help="Weight Decay.")
    
    parser.add_argument('--epochs', type=int, help='Maximum number of epochs')
    parser.add_argument('--warmup', type=int, help='Number of warmup epochs')
    parser.add_argument("--nw",     type=int, help="Numbers of workers.")

    parser.add_argument('--backbone',   type=str, help='Backbone model name [resnet50, dinov2]')
    parser.add_argument('--unfreeze_n', type=int, help='Number of blocks to unfreeze in the backbone.')
    parser.add_argument("--dim",        type=int, help="Output dimensionality.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    hparams = HyperParams()
    
    if args.seed:
        seed = args.seed
    if args.compile:
        hparams.compile = True
    if args.bs:
        hparams.batch_size = args.bs
    if args.lr:
        hparams.lr = args.lr
    if args.wd:
        hparams.weight_decay = args.wd
    if args.epochs:
        hparams.max_epochs = args.epochs
    if args.warmup:
        hparams.warmup_epochs = args.warmup
    if args.nw:
        hparams.num_workers = args.nw
    if args.backbone:
        hparams.backbone_name = args.backbone
    if args.unfreeze_n:
        hparams.unfreeze_n_blocks = args.unfreeze_n
    if args.dim:
        hparams.output_dim = args.dim
    
    train(hparams, dev_mode=args.dev)