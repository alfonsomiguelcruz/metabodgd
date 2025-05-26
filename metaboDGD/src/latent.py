import torch
import torch.nn as nn

## TODO: Add Gaussian and Softball prior classes
class GaussianMixtureModel(nn.Module):
    def __init__(self):
        super(GaussianMixtureModel, self).__init__()

    
    def forward(self, z):
        pass

class RepresentationLayer(nn.Module):
    def __init__(
        self,
        latent_dim,
        n_sample,
        values=None
    ):
        super(RepresentationLayer, self).__init__()

        # If values were not provided
        if values is None:
            self.dim = latent_dim
            self.n_sample = n_sample
            self.z = nn.Parameter(
                torch.normal(0,
                             1,
                             size=(self.n_sample, self.dim),
                             requires_grad=True)
            )
        # If values were provided
        else:
            self.dim = values.shape[-1]
            self.n_sample = values.shape[0]
            self.z = nn.Parameter(
                torch.zeros_like(values),
                requires_grad=True
            )
            
            with torch.no_grad():
                self.z.copy_(values)

    def forward(self, idx=None):
        if idx is None:
            return self.z
        else:
            return self.z[idx]