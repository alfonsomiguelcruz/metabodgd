import torch
import math 

class GaussianPrior:
    """
    Class implementing a Gaussian prior distribution.

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent representation.

    mean : `float`
        Initialized mean of the normal distribution.

    stddev : `float`
        Initialized standard deviation of the normal
        distribution.
    """

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
        """
        Get a number of samples with the normal
        distribution.

        Parameters
        ----------
        n_sample : `int`
            The number of samples to get from the
            distribution.


        Returns
        -------
        samples : `torch.Tensor`
            A tensor containing the sampled points
            from the distribution.
        """

        return self.gaussian_dist.sample((n_sample, self.dim))
    

    def log_prob(self, x):
        """
        Gets the log-probability of observing a datapoint
        in the normal distribution.

        Parameters
        ----------
        x : `torch.Tensor`
            A tensor representing a datapoint.


        Returns
        -------
        log-probability : `torch.Tensor`
            A tensor containing the log-probabilities of
            observing the datapoint in the distribution.
        """

        return self.gaussian_dist.log_prob(x)
    


class SoftballPrior:
    """
    Class implementing a Softball prior distribution.

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent representation.

    radius : `int`
        Initialized radius of the Softball distribution.

    sharpness : `int`
        Initialized sharpness of the Softball
        distribution.
    """

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
        """
        Get a number of samples with the Softball
        distribution.

        Parameters
        ----------
        n_sample : `int`
            The number of samples to get from the
            distribution.


        Returns
        -------
        samples : `torch.Tensor`
            A tensor containing the sampled points
            from the distribution.
        """

        with torch.no_grad():
            sample = torch.randn((n_sample, self.dim))
            
            sample.div_(sample.norm(dim=-1, keepdim=True))

            local_len = self.radius * torch.pow(torch.rand((n_sample, 1)), 1.0 / self.dim)

            sample.mul_(local_len.expand(-1, self.dim))
        
        return sample
    

    def log_prob(self, x):
        """
        Gets the log-probability of observing a datapoint
        in the Softball distribution.

        Parameters
        ----------
        x : `torch.Tensor`
            A tensor representing a datapoint.


        Returns
        -------
        log-probability : `torch.Tensor`
            A tensor containing the log-probabilities of
            observing the datapoint in the distribution.
        """
        
        norm = math.lgamma(1 + self.dim * 0.5)   - \
               self.dim * (math.log(self.radius) + \
               0.5 * math.log(math.pi))
        
        return (norm - torch.log(
            1 + torch.exp(self.sharpness * (x.norm(dim=-1) / self.radius - 1))
        ))