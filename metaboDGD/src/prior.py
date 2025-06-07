import torch
import math 

class GaussianPrior:
    def __init__(
        self,
        latent_dim,
        mean,
        stddev
    ):
        self.dim  = latent_dim
        self.mean = mean
        self.stddev = stddev
        self.gaussian_dist = torch.distributions.normal.Normal(mean, stddev)

    
    def sample(self, n_sample):
        return self.gaussian_dist.sample((n_sample, self.dim))
    

    def log_prob(self, x):
        return self.gaussian_dist.log_prob(x)
    


class SoftballPrior:
    def __init__(
        self,
        latent_dim,
        radius,
        sharpness
    ):
        self.dim = latent_dim
        self.radius = radius
        self.sharpness = sharpness
    
    def sample(self, n_sample):
        with torch.no_grad():
            sample = torch.randn((n_sample, self.dim))
            
            sample.div_(sample.norm(dim=-1, keepdim=True))

            local_len = self.radius * torch.pow(torch.rand((n_sample, 1)), 1.0 / self.dim)

            sample.mul_(local_len.expand(-1, self.dim))
        
        return sample
    

    def log_prob(self, x):
        norm = math.lgamma(1 + self.dim * 0.5)   - \
               self.dim * (math.log(self.radius) + \
               0.5 * math.log(math.pi))
        
        return (norm - torch.log(
            1 + torch.exp(self.sharpness * (x.norm(dim=-1) / self.radius - 1))
        ))