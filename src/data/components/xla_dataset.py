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
from aug.wrapper_v2 import Augmenter

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
        self.data_dir = data_dir
        print(self.data_dir)
        self.vocab = None 
        self.task = task
        self.samples = self.load_data(manifest)
        self.id2sample = None
        if task == "val":
            self.vocab = None

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index) -> tuple:
        assert self.vocab is not None

        filename, label = self.samples[index]

        # open & process image
        image_path = self.data_dir + filename
        
        image = np.array(Image.open(image_path).convert("RGB"))

        return {'filename': filename, 'image': image, 'label': self.vocab[label]}

    def load_data(self, manifest):
        samples = []
        with open(manifest, "r") as file:
            self.id2sample = json.load(file)[self.task]
        keys = sum([[id] * len(self.id2sample[id]) for id in self.id2sample.keys()], [])
        self.vocab = dict(zip(keys.copy(), range(len(keys))))
        filenames = sum(list(self.id2sample.values()), [])
        sample2id = list(zip(filenames, keys))

        return sample2id
    
    def __len__(self):
        return len(self.samples)

class XLATransformedDataset(Dataset):
    def __init__(
            self,
            dataset: XLADataset, 
            augmenter: Augmenter,
            transform: transforms.Compose,
            p = [0.6, 0.4]
    ):
        assert augmenter is not None
        self.dataset = dataset
        # shape transformation
        self.augmenter = augmenter

        # pixel transform
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    


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
    train_data_dir = "./data"
    manifest = "./data/wb_recognition_dataset/manifest.json"

    dataset = OCRDataset(data_dir=train_data_dir, manifest=manifest)
    img = dataset[0]
    print(img['image'].shape)