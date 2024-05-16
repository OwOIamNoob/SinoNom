from typing import Any, Dict, Optional, Tuple

import torch
import rootutils
import copy
rootutils.setup_root(search_from=__file__, indicator="setup.py", pythonpath=True)
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from src.data.components.vietocr_aug import ImgAugTransform
# from src.data.components.ocr_vocab import Vocab
from src.data.components.aug.wrapper_v2 import Augmenter
from src.data.components.xla_dataset import XLADataset, XLATransformedDataset
from src.data.components.xla_utils import XLARandomSampler, XLACollator
import datetime
import shutil
import os 

class XLADataModule(LightningDataModule):
    def __init__(   self,
                    data_dir,
                    manifest,
                    base_augmenter: Augmenter = Augmenter(),
                    color_augmenter: ImgAugTransform = ImgAugTransform(0.3),
                    image_shape: list = [64, 64],
                    batch_size: int = 8,
                    num_workers: int = 0,
                    pin_memory: bool = False, 
                    shuffle: bool = True,
                    upsampling: bool = True, 
                    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        # self.data_test: Optional[Dataset] = None

        # self.batch_size_per_device = batch_size
        
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None

        # self.val_loader: Optional[DataLoader] = None
    
    def prepare_data(self) -> None:
        pass 
    
    def setup(self):
        if not self.data_train or not self.data_val:
            train_dataset = XLADataset(data_dir=self.hparams.data_dir,
                                        manifest= self.hparams.manifest,
                                        task= 'train'
                                        )
            val_dataset = XLADataset(data_dir=self.hparams.data_dir,
                                        manifest= self.hparams.manifest,
                                        task= 'val')
            val_dataset.vocab = train_dataset.vocab

            # Specific augment
            train_aug = copy.deepcopy(self.hparams.base_augmenter)
            train_aug.task = 'train'

            val_aug = copy.deepcopy(self.hparams.base_augmenter)
            val_aug.task = 'val'

            self.data_train = XLATransformedDataset(dataset=train_dataset,
                                                    augmenter=train_aug,
                                                    transform=self.hparams.color_augmenter,
                                                    p = [0.6, 0.2, 0.2])

            self.data_val = XLATransformedDataset(  dataset=val_dataset, 
                                                    augmenter=val_aug,
                                                    transform=None, 
                                                    p = [1, 0, 0]) 
        return True 

    def train_dataloader(self):
        # if isinstance(self.train_dataloader, DataLoader):
        #     return self.train_dataloader
        
        train_sampler = XLARandomSampler( data_source=self.data_train.dataset.ranges,
                                           max_size=len(self.data_train),
                                           shuffle=self.hparams.shuffle,
                                           balance=self.hparams.upsampling)

        collator = XLACollator( num_class=self.data_train.num_classes(),
                                image_shape=self.hparams.image_shape)

        self.train_loader =  DataLoader(self.data_train,
                            batch_size = self.hparams.batch_size,
                            shuffle=False,
                            sampler=train_sampler,
                            num_workers=self.hparams.num_workers,
                            collate_fn=collator,
                            pin_memory= self.hparams.pin_memory
                            )
        # print(self.train_dataloader)
        
        return self.train_loader
    
    def val_dataloader(self) -> DataLoader[Any]:
        if self.val_loader is not None:
            return self.val_loader
        
        self.val_loader = DataLoader(dataset=self.data_val,
                                    batch_size=self.hparams.batch_size,
                                    shuffle=False,
                                    num_workers=1,
                                    collate_fn=XLACollator(num_class=self.data_val.num_classes(),
                                                        image_shape=self.hparams.image_shape))
        return self.val_loader

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

if __name__ == "__main__":
    data_dir = "/data/hpc/potato/sinonom/data/wb_recognition_dataset/"
    manifest = "/data/hpc/potato/sinonom/data/wb_recognition_dataset/manifest_split.json"
    print(os.path.exists(manifest))
    augmenter = Augmenter(  texture_path="/data/hpc/potato/sinonom/data/augment/background/base/", 
                            bg_checkpoint="/data/hpc/potato/sinonom/data/augment/background/",
                            task="train")
    
    transform = ImgAugTransform(0.3)

    datamodule = XLADataModule(data_dir,
                                manifest,
                                base_augmenter=augmenter,
                                color_augmenter=transform,
                                image_shape = [64, 64],
                                batch_size=64,
                                num_workers= 0,
                                pin_memory=False,
                                shuffle = False,
                                upsampling = False)
    
    datamodule.setup()
    # datamodule.prepare_data()
    # print(type(datamodule))
    # print(datamodule)
    loader = datamodule.train_dataloader()
    # dataset = datamodule.data_val
    # sample = dataset[0]
    # ranges = dataset.dataset.ranges 
    # print(ranges)
    it = iter(loader)
    m = 0
    while m < 20000:
        m += 1
        
        sample = next(it)
        print(sample['labels'])