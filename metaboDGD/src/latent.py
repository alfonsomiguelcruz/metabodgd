import torch
import math
import torch.nn as nn
from metaboDGD.src.prior import SoftballPrior, GaussianPrior

class GaussianMixtureModel(nn.Module):
    """
    Class implementing the Gaussian mixture model
    component of the deep generative decoder model.

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent representation.

    n_comp : int
        Number of GMM components.

    cm_type : str, default="diagonal"
        Type of covariance matrix used in the GMM.

    softball_radius : int, default=5
        Radius parameter of the softball prior.

    softball_sharpness : int, default=5
        Sharpness parameter of the softball prior.

    gaussian_mean : float, default=-5.0
        Mean parameter of the Gaussian prior.

    gaussian_stddev : float, default=0.5
        Standard deviation parameter of the Gaussian prior.

    dirichlet_alpha : int, default=11
        Alpha parameter of the Dirichlet prior.
    """

    def __init__(
            self,
            latent_dim,
            n_comp,
            cm_type='diagonal',
            softball_radius=5,
            softball_sharpness=5,
            gaussian_mean=-5.0,
            gaussian_stddev=0.5,
            dirichlet_alpha=11,
        ):
        super().__init__()

        # Latent dimension
        self.dim = latent_dim

        # Number of GMM components
        self.n_comp = n_comp

        # Type of covariance matrix
        self.cm_type=cm_type

        # Alpha parameter for the Dirichlet prior
        self.alpha = dirichlet_alpha

        # Softball prior for the GMM means
        self.means_prior = {
            'dist': SoftballPrior(
                latent_dim=self.dim,
                radius=softball_radius,
                sharpness=softball_sharpness
            )
        }

        # Gaussian prior for the GMM log variances
        self.log_var_prior = {
            'dist': GaussianPrior(
                latent_dim=self.dim,
                mean=gaussian_mean,
                stddev=gaussian_stddev
            )
        }

        # GMM component log variance parameters
        self.log_var  = nn.Parameter(
                            self.log_var_prior['dist'].sample(n_sample=n_comp),
                            # torch.full(size=(self.n_comp, self.dim), fill_value=gaussian_mean),
                            requires_grad=True
                        )
        
        # GMM component mean parameters
        self.means    = nn.Parameter(
                           self.means_prior['dist'].sample(n_sample=self.n_comp),
                           requires_grad=True
                        )
        
        # GMM component weight parameters
        self.weights  = nn.Parameter(
                            torch.ones(self.n_comp),
                            requires_grad=True
                        )


    def get_log_prob_comp(self, x):
        """
        Get the per-component log-probability of every
        data point.

        Parameters
        ----------
        x : `torch.Tensor`
            A tensor containing the latent representations.


        Returns
        -------
        log_prob : `torch.Tensor`
            A tensor of the per-component log-probability of
            every data point.
        """

        pi_term = - 0.5 * self.dim * math.log(2 * math.pi)

        ## Temporarily set log_var_prior factor for diagonal covariance matrices
        log_var_prior = 0.5
        cm_dependent_term = - (log_var_prior * self.log_var.sum(-1))

        mean_term = -(x.unsqueeze(-2) - self.means).square().div(\
                    2 * torch.exp(self.log_var)).sum(-1)

        log_prob = pi_term + cm_dependent_term + mean_term
 
        log_prob += torch.log_softmax(self.weights, dim=0)

        return log_prob


    def get_mixture_probs(self):
        """
        Get the probabilities of the GMM components from
        the component weights.

        Returns
        -------
        component probabilities : `torch.Tensor`
            A tensor of the GMM component probabilities.
        """
        return torch.softmax(self.weights, dim=-1)


    def get_prior_log_prob(self):
        """
        Get the log-probability of the GMM parameters.

        Returns
        -------
        p : `float`
            A float representing the log-probability of
            the prior over the GMM parameters (means, log-
            variance, and weights).
        """
        p = 0.0

        # Assume weights prior is Dirichlet (add alpha, constant)
        # self.alpha = 11
        p = math.lgamma(self.n_comp * self.alpha) - \
            self.n_comp * math.lgamma(self.alpha)
        
        if self.alpha != 1:
            p += (self.alpha - 1.0) * (self.get_mixture_probs().log().sum())
        

        # Assume means prior is Softball
        p += self.means_prior['dist'].log_prob(self.means).sum()

        # Assume logvar prior is Gaussian
        p += self.log_var_prior['dist'].log_prob(self.log_var).sum()

        return p


    def forward(self, x):
        """
        Get the absolute log-probability density for a set
        of representations.

        Parameters
        ----------
        x : `torch.Tensor`
            A tensor containing the latent representations.


        Returns
        -------
        y : `torch.Tensor`
            A tensor of the absolute log-probability
            densities of every representation.
        """

        y = self.get_log_prob_comp(x)

        y = torch.logsumexp(y, dim=-1)

        y = y + self.get_prior_log_prob()

        return y
    

    def sample_new_points(self, n_samples):
        """
        Get new points from the GMM.

        Parameters
        ----------
        n_samples : `int`
            The number of samples to be taken
            from the GMM.


        Returns
        -------
        sampled points : `torch.Tensor`
            A tensor of `n_samples` copies of the
            GMM component means.
        """

        with torch.no_grad():
            sampled_points = torch.repeat_interleave(self.means.clone().detach().unsqueeze(0),
                                                     n_samples,
                                                     dim=0)
        
        return sampled_points.view(n_samples * self.n_comp, self.dim)



class RepresentationLayer(nn.Module):
    """
    Class implementing the representation layer
    for learning the latent representations and
    accumulating gradients.

    Parameters
    ----------
    values : `torch.Tensor`, default=None
        A tensor to initialize the representations in
        the layer.
    """

    def __init__(
        self,
        values=None
    ):
        super().__init__()

        # If values were not provided
        if values is None:
            ## TODO: TO FIX (Add yaml file for parameters)
            self.dim = -1
            self.n_sample = -1
            self.z = -1
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
        """
        Access the representations in the layer.

        Parameters
        ----------
        idx : `int`, default=None
            The index of the representation
            being accessed.


        Returns
        -------
        z : `torch.Tensor`
            A tensor of representations in the layer.
        """

        if idx is None:
            return self.z
        else:
            return self.z[idx]