import torch
# from torch.utils.data import Dataset

class MetaboliteAbundanceDataset(object):
    def __init__(self, np_mat):
        self.metabolite_abundances = np_mat
        
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx=None):
        pass