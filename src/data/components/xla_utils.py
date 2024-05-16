import cv2
import numpy as np 
import torch 
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
import random


class XLARandomSampler(Sampler):
    def __init__(
        self, 
        data_source, 
        max_size = 0,
        shuffle = True,
        upsample = None,
        balance = False 
    ):
        self.data_source = data_source
        self.shuffle = shuffle
        self.upsample_profile = upsample
        self.balance = balance
        self.total = 0
        self.max_size = max_size

    def __iter__(self):
        
        if not self.upsample_profile:
            if self.balance:
                self.upsample_profile = self.upsample()

        lst = []
        if self.upsample_profile:
            lst = self.get_ids(self.upsample_profile)
        else: 
            lst = range(self.max_size)
        
        self.total = len(lst)
        
        if self.shuffle:
            random.shuffle(lst)

        return iter(lst)
    
    # mitigate sample unbalance by square root.
    def upsample(self):
        samples = [h[1] - h[0] for h in list(self.data_source.values())]
        mean = np.mean(samples)
        ratio = np.sqrt(np.array(samples) / mean)
        subset = np.ceil(ratio * mean).astype(int) 
        return dict(zip(list(self.data_source.keys()), subset.tolist()))

    def __len__(self):
        return self.total 

    def flatten_list(self, lst):
        return [item for sublist in lst for item in sublist]

    def get_ids(self, request):
        ids = []
        for key in request.keys():
            assert key in self.data_source.keys(), "Key not found"
            lr = self.data_source[key]
            key_ids = np.random.randint(lr[0], lr[1] + 1,  size=(request[key], )).tolist()
            # print(key_ids)
            ids += key_ids 
        
        return ids

        

class XLACollator(object):
    def __init__(self, num_class, image_shape=[64, 64]):
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.num_class = num_class
        self.image_shape = image_shape

    def __call__(self, batch):
        filenames = []
        imgs = []
        labels = []
        for sample in batch:
            filename = sample['filename']
            image = sample['image']
            label = sample['label']

            filenames.append(filename)
            imgs.append(self.transform(cv2.resize(  image, 
                                                    self.image_shape, 
                                                    interpolation=cv2.INTER_LINEAR)))
            # print("Label:", label)
            labels.append(int(label))

        print(labels)
        rs = {
            'imgs': torch.stack(imgs, dim=0),
            'filenames': filenames,
            'labels': torch.nn.functional.one_hot(  torch.Tensor(labels).to(torch.int64), 
                                                    num_classes=self.num_class)
        }   
        
        return rs