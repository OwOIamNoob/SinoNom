import json
import os 
import numpy as np

class Analyzer:
    def __init__(self, manifest_path):
        with open(manifest_path, "r") as file:
            self.dataset = json.load(file)
        
        self.train_dist = dict()
        self.val_dist = dict()
    
    def prepare(self):
        train_dataset = self.dataset['train']
        keys = list(train_dataset.keys())
        train_dist = [len(data) for data in train_dataset.values()]
        self.train_dist = dict(zip(keys, train_dist))
        
        val_dataset = self.dataset['val']
        val_keys = list(val_dataset.keys())
        self.val_dist = dict(zip(keys, np.zeros(len(keys), dtype=int).tolist()))
        for key in val_keys:
            self.val_dist[key] += len(val_dataset[key])
