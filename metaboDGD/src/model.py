import torch
from metaboDGD.src.decoder import Decoder
from metaboDGD.src.latent  import GaussianMixtureModel

class MetaboDGD():
    # TODO:
    # Create the initialization of parameters based on bulkDGD
    # (dim, n_comp, etc.)
    def __init__ (self):
        super(MetaboDGD, self).__init__()

        self.dec = Decoder()
        self.gmm = GaussianMixtureModel()
