from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
# from src.data.components.vietocr_aug import ImgAugTransform
# from src.data.components.ocr_vocab import Vocab
from torch.utils.data.sampler import Sampler
import random
import torch
import cv2
import numpy as np
# from src.data.components.custom_aug.wrapper import Augmenter
import math
import shutil
import json

from src.data.components.aug.wrapper_v2 import Augmenter
from src.data.components.vietocr_aug import ImgAugTransform




def delete_contents_of_folder(folder_path):
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        print(f"Deleted all contents of the folder: {folder_path}")
    except Exception as e:
        print(f"Error deleting contents of the folder: {folder_path}")




class XLADataset(Dataset):
    ''' This dataset only loads images from files into numpy arrays '''

    def __init__(
        self, 
        data_dir: str, 
        manifest: str,
        task: str
    ):
        super().__init__()
        self.data_dir = data_dir
        print(self.data_dir)
        self.vocab = None 
        self.task = task
        self.ranges = None
        self.id2sample = None
        self.samples = self.load_data(manifest)
        
        if task == "val":
            self.vocab = None

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index) -> tuple:
        assert self.vocab is not None
        assert index < len(self), "{1} is out of index of {2}".format(index, len(self))
        filename, label = self.samples[index]
        # print("data: ", filename, label)
        # open & process image
        image_path = os.path.join(self.data_dir, self.task)
        image_path = os.path.join(image_path, filename)
        
        image = np.array(Image.open(image_path).convert("RGB"))

        return {'filename': filename, 'image': image, 'label': label}

    def load_data(self, manifest):
        samples = []
        with open(manifest, "r") as file:
            self.id2sample = json.load(file)[self.task]
        
        keys = list(self.id2sample.keys())
        # print("Keys range:", np.max(keys), np.min(keys))
        self.vocab = dict(zip(keys, range(len(keys))))
        
        key_num = [len(self.id2sample[id]) for id in list(self.id2sample.keys())]
        ranges = np.cumsum(key_num).tolist() + [0]
        key_ranges = [[ranges[i-1], ranges[i]] for i in range(0, len(ranges) - 1)]
        # get id distribution of space for upsampling 
        self.ranges = dict(zip(keys, key_ranges))
        print("total dataset: ", key_ranges[-2])
        # prepares token 
        sample2id = []

        for key in keys: 
            filenames = self.id2sample[key]
            for fname in filenames:
                sample2id.append([fname, self.vocab[key]])   
        # # filenames = sum(list(self.id2sample.values()), [])
        # # sample_keys = sum([[id] * len(self.id2sample[id]) for id in self.id2sample.keys()], [])
        
        # sample2id = list(zip(filenames, sample_keys))
        print("Actual samples", len(sample2id))
        return sample2id
    
    def __len__(self):
        return len(self.samples)
    
    def num_classes(self):
        return len(self.vocab)

class XLATransformedDataset(Dataset):
    def __init__(
            self,
            dataset: XLADataset, 
            augmenter: Augmenter,
            transform: ImgAugTransform,
            p = [0.6, 0.2, 0.2]
    ):
        assert augmenter is not None
        self.dataset = dataset
        # shape transformation
        self.augmenter = augmenter
        self.augmenter.task = self.dataset.task

        # pixel transform
        self.transform = transform
        self.p = p

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        sample = self.dataset[index]
        filename = sample['filename']
        image = sample['image']
        label = sample['label']
        print("data: ", filename, label)
        # first of all, transform the image before applying 
        image = self.augmenter.full_augment(image, 
                                            choice=self.p,
                                            fname=filename,
                                            borderMode='native')
        if isinstance(self.transform, ImgAugTransform):
            image = self.transform(image)
        
        return {'filename': filename, 'image': image, 'label': label}

    def num_classes(self):
        return self.dataset.num_classes()
    


# class OCRTransformedDataset(Dataset):
#     ''' This dataset applies all custom transformations & augmentation to input images and encodes labels '''
#     def __init__(
#         self, 
#         dataset: OCRDataset,
#         task: str,
#         images_epoch_folder_name: str,
#         vocab = Vocab(),
#         custom_augmenter = Augmenter(),
#         p = [0.6, 0.2, 0.1, 0.1],
#     ):
#         self.dataset = dataset
#         self.vocab = vocab
#         self.task = task
#         self.images_epoch_folder_name = images_epoch_folder_name

#         self.custom_augmenter = custom_augmenter
#         self.p = p
        
#         delete_contents_of_folder(f"aug_epoch/{self.images_epoch_folder_name}/{self.task}")

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, index):
#         sample = self.dataset[index]
#         filename = sample['filename']
#         image = sample['image']
#         word = sample['label']

#         # process image
#         try:
#             os.mkdir(f"aug_epoch/{self.images_epoch_folder_name}")
#             os.mkdir(f"aug_epoch/{self.images_epoch_folder_name}/{self.task}")
#         except FileExistsError:
#             pass
        
#         folder_path = f"aug_epoch/{self.images_epoch_folder_name}/{self.task}"
#         try:
#             os.mkdir(folder_path)
#         except FileExistsError:
#             pass

#         image_path = os.path.join(folder_path, f"{index}_{filename.strip().split('.')[0]}.png")
#         if not os.path.exists(image_path):
#             if self.custom_augmenter:
#                 image: np.array = self.custom_augmenter(image, word, 1, self.p)[0]
#             cv2.imwrite(image_path, image)

#         # encoding word
#         label = self.vocab.encode(word)

#         return {'filename': filename, 'image_path': image_path, 'label': label}

if __name__ == "__main__":
    import rootutils
    rootutils.setup_root(search_from=__file__, indicator="pyproject.toml", pythonpath=True)
    data_dir = "/data/hpc/potato/sinonom/data/wb_recognition_dataset/"
    manifest = "/data/hpc/potato/sinonom/data/wb_recognition_dataset/manifest_split.json"

    dataset = XLADataset(data_dir=data_dir, manifest=manifest, task="train")
    img = dataset[0]
    print(img['image'].shape)

    augmenter = Augmenter(  texture_path="/data/hpc/potato/sinonom/data/augment/background/base/", 
                            bg_checkpoint="/data/hpc/potato/sinonom/data/augment/background/",
                            task="train")
    
    transform = ImgAugTransform(0.3)

    transformed_dataset = XLATransformedDataset(dataset=dataset,
                                                augmenter=augmenter,
                                                transform=transform)
    
    sample = transformed_dataset[100]
    print(sample['filename'])
    print(sample['image'].shape)