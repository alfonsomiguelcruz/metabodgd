from torch.utils.data import Dataset

class MetaboliteDataset(Dataset):
    """
    Class implementing a dataset containing metabolite
    abundance data of multiple samples, implemented as
    a `Dataset` object.

    Parameters
    ----------
    np_mat : numpy.array (n_samples, n_metabolites)
        Matrix of metabolite abundances, where
        `n_samples` is the number of samples and
        `n_metabolites` is the number of metabolites.
    
    cohort_labels : list (n_samples,)
        List of labels for each sample.
    """


    def __init__(self,
                 np_mat,
                 cohort_labels):
        """
        Initializes the dataset object.
        """

        # Metabolite abundance matrix 
        self.metabolite_abundances = np_mat

        # Number of metabolites
        self.n_metabolites = self.metabolite_abundances.shape[1]

        # Sample labels
        self.cohort_labels = cohort_labels


    def __len__(self):
        """
        Gets the number of samples in the dataset.
        """

        return self.metabolite_abundances.shape[0]


    def __getitem__(self, idx):
        """
        Gets the sample in the dataset at index i.
        """

        return self.metabolite_abundances[idx], idx
    

    def get_labels(self, idx=None):
        """
        Gets the labels of the samples in dataset.
        """

        if idx is None:
            return self.cohort_labels
        else:
            return self.cohort_labels[idx]