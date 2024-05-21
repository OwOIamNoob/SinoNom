import json
import os 
import numpy as np
import shutil

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


class Analyzer:
    def __init__(self, 
                manifest_path, 
                cache_dir = "cache",
                data_dir = "/data/hpc/"):
        
        self.dataset = None

        self.train_full_dist = dict()
        self.train_hard_dist = dict()
        self.val_dist = dict()

        # after setup, we will use dizz nuts
        self.train_hard = dict()
        self.train = dict()
        self.val = dict()

        # cache new state of dataset in case algorithm gone wrong :) 
        self.new_train = None
        self.new_train_hard = None
        self.cache = None
        self.cache_dist = None

        # path for cp command
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        self.total = 0
        self.prepare(manifest_path)

    def prepare(self, manifest_path):
        """ Prepare sample sets for processing
        """
        with open(manifest_path, "r") as file:
            dataset = json.load(file)
            self.dataset = dataset
        train_dataset = dataset['train']
        keys = list(train_dataset.keys())
        train_full_dist = [len(data) for data in train_dataset.values()]
        self.train_full_dist = dict(zip(keys, train_full_dist))
        
        val_dataset = dataset['val']
        val_keys = list(val_dataset.keys())
        self.val_dist = dict(zip(keys, np.zeros(len(keys), dtype=int).tolist()))
        

        # train samples that can transfer to valid set 
        for key in keys:
            i = 0
            self.train_hard[key] = []
            for index in range(len(train_dataset[key])):
                index -= i
                if 'nlvnpf' in train_dataset[key][index]:
                    i += 1
                # transfer hard dataset to other dataset 
                    self.train_hard[key].append(train_dataset[key].pop(index))
                
            self.train_hard_dist[key] = i
    

        for key in val_keys:
            self.val_dist[key] += len(val_dataset[key])
        
        self.train = train_dataset
        self.val = dataset['val']
        
        # load cache, cache included means new state :D
        if 'cache' in dataset.keys():
            self.cache = dataset['cache']
            self.cache_dist = dict()
            for key in keys:
                    self.cache_dist[key] = len(self.cache[key]) if key in self.cache.keys() else 0
        
        # To manage dataset description.
        if self.total == 0:
            self.total = np.sum(list(self.train_full_dist.values())) + np.sum(list(self.val_dist.values()))
            if isinstance(self.cache, dict):
                self.total += np.sum([len(label) for label in self.cache.values()])
            print("Total samples:", self.total)
        else: 
            ref = np.sum(list(self.train_full_dist.values())) + np.sum(list(self.val_dist.values()))
            if isinstance(self.cache, dict):
                ref += np.sum([len(label) for label in self.cache.values()])
            print("Total samples differences:", ref - self.total)

    def valid_fill_label(self, alpha=0.8):
        """ This function analyze the missing labels, also label unbalance between train and val
            Thus, propose a new manifest that make sure every labels have its sample in validation set"""
        # backward folder path
        v_dist = np.array(list(self.val_dist.values()))
        if self.cache_dist:
            v_dist += np.array(list(self.cache_dist.values()))
        
        v_mean = np.mean(v_dist[v_dist > 0])
        print("Mean:", v_mean)
        
        v_std = np.sqrt(np.var(v_dist[v_dist != 0]))
        

        # get missing label indexes
        v_missing = (v_dist < 1).astype(int)
        v_deviation = v_dist - v_mean
        v_upsample = v_deviation

        # Exclude all missing samples, suffient sample set.
        v_upsample[(np.abs(v_deviation) <= v_std / alpha) | 
                    (v_deviation > 0) | (v_missing > 0)] = 0
        
        # Acceptable required no. samples for missing labels
        # Also rounding up fundamentals
        # Cannot make it out of acceptable range in case it snaps to 0 :)
        # Can only make up biased balancing strategies  
        v_missing = np.ceil(v_missing.astype(int) * v_mean).astype(int)
        v_upsample = np.ceil(v_upsample).astype(int)
        
        keys = list(self.train_full_dist.keys())
        print("Valid mean samples", v_mean)
        print("Valid standard deviation", v_std)
        print("Missing labels", np.sum(v_missing))

        return v_mean, v_std, v_dist, dict(zip(keys, v_upsample)), dict(zip(keys, v_missing))
        
    def query_train(self, query, maximum_exploit=0.4):
        """ Retrieve training examples as valid label filling...
            
            Args:
                query: samples query with required sample quantities from training set
                maximum_exploit: Maximum percentage of samples can be taken from training hard examples
        """
        # Being sures that all dictionary share the same key sets,
        # Else this algorithm will break

        assert list(query.keys()) == list(self.train_hard_dist.keys())
        train_h_dist = np.array(list(self.train_hard_dist.values()))
        train_dist = np.array(list(self.train_full_dist.values())) - train_h_dist
        
        v_req = np.array(list(query.values()))
        
        train_h_avail = np.floor(train_h_dist * maximum_exploit).astype(int)
        
        train_avail = np.floor(train_dist * maximum_exploit).astype(int)
        
        # determine how many sample in nlvnpf dataset that valid can take
        v_h_req = np.min(np.stack([v_req, train_h_avail], axis=1), axis=1)
        
        # in other case, take from ordinary samples - Thuong samples
        v_n_req = np.min(np.stack([v_req - v_h_req, train_avail], axis=1), axis=1)
        
        return  dict(zip(list(query.keys()), v_req.tolist())), dict(zip(list(query.keys()), v_h_req.tolist())), dict(zip(list(query.keys()), v_n_req.tolist()))


    def extract(self, req, h_req, n_req, maximum_exploit=0.4, export=None, refresh=False):
        """ This function diversify the validation set by taking samples from training set. 
        """
        if refresh is True:
            delete_contents_of_folder(self.data_dir + "/cache")
            os.mkdir(self.data_dir + "/cache/copy")
            self.new_train = self.train.copy()
            self.new_train_hard = self.train_hard.copy()

        for key in list(self.new_train.keys()):
            if req[key] == 0:
                continue

            # make sure that cache also have same keys with train set    
            if key not in self.cache.keys():
                self.cache[key] = []

            for i in range(h_req[key]):
                index = np.random.choice(range(len(self.new_train_hard[key])))
                data = self.new_train_hard[key].pop(index)
                fname = data.replace("/", "_")
                self.cache[key].append(fname)
                shutil.copy(self.data_dir + "/train/" + data, self.data_dir + "/cache/" + fname)

            for i in range(n_req[key]):
                index = np.random.choice(range(len(self.new_train[key])))
                data = self.new_train[key].pop(index)
                fname = data.replace("/", "_")
                self.cache[key].append(fname)
                shutil.copy(self.data_dir + "/train/" + data, self.data_dir + "/cache/" + fname)
            
            # in case that all request violated the exploit limit
            # only for missing labels  and demanding samples
            if len(self.cache[key]) == 0 and refresh is True and req[key] != 0:
                print("Insufficient dataset, undergo copying samples label {}".format(key))

                if self.train_full_dist[key] <= np.ceil(1 / maximum_exploit):
                    index =  np.random.choice(range(self.train_full_dist[key]))
                    data = self.dataset['train'][key][index]
                    fname = "copy/" + data.replace("/", "_")
                    self.cache[key].append(fname)
                    shutil.copy(self.data_dir + "/train/" + data, self.data_dir + "/cache/" + fname)

        
        if isinstance(export, str):
            output = dict()
            output['cache'] = self.cache.copy()
            output['train'] = self.new_train.copy()

            # merge train dataset back together
            for key in self.new_train.keys():
                output['train'][key] += self.new_train_hard[key]
            
            output['val'] = self.val
            # json = json.dumps(output, indent=4)
            with open(export, "w") as f:
                json.dump(output, f, indent=4)
                f.close()

            # Reload dataset
            self.prepare(export)

    def train_loss_weight(self):
        assert self.val_dist.keys() != len(self.train_full_dist.keys()), "Validation set label vanished or overtaken or misordered from training set !" 
        
        # v_d = np.array([len(value) for value in self.val.values()]) + np.array([len(value) for value in self.cache.values()])
        t_d = np.array(list(self.train_full_dist.values()))
        ratio = np.sqrt(np.mean(t_d) / t_d)

        return ratio
        
    def merge_cache_val(self, export = None):
        # prefix = "../cache"
        for key in self.cache.keys():
            if key not in self.val.keys():
                self.val[key] = []
            
            for sample in self.cache[key]:
                full_path = os.path.join(self.cache_dir, sample)
                output_path = full_path.replace("/", "_")
                self.val[key].append(output_path)
                print(output_path)
                shutil.copy(os.path.join(self.data_dir, full_path), os.path.join(self.data_dir + "/val", output_path) )

        if isinstance(export, str):
            output = dict()
            output['train'] = self.train.copy()

            # merge train dataset back together
            for key in self.train.keys():
                output['train'][key] += self.train_hard[key]
            
            output['val'] = self.val
            
            with open(export, "w") as f:
                json.dump(output, f, indent=4)
                f.close()

            # Reload dataset
            self.prepare(export)

            
                
    # def query_train(self, )