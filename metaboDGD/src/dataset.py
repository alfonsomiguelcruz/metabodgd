from torch.utils.data import Dataset

class MetaboliteDataset(Dataset):
    def __init__(self, np_mat, cohort_labels):
        # Instantiate Dataset object
        # Initialize Directory containing data, annotations, transforms
        self.metabolite_abundances = np_mat
        # self.metabolite_abundances = self.metabolite_abundances.to(torch.float32)
        self.n_metabolites = self.metabolite_abundances.shape[1]

        self.cohort_labels = cohort_labels

    def __len__(self):
        return self.metabolite_abundances.shape[0]

    def __getitem__(self, idx):
        return self.metabolite_abundances[idx], idx
    
    def get_labels(self, idx=None):
        if idx is None:
            return self.cohort_labels
        else:
            return self.cohort_labels[idx]