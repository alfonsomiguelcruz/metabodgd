import torch
import math
import torch.nn as nn
from metaboDGD.src.prior import SoftballPrior, GaussianPrior

class GaussianMixtureModel(nn.Module):
    def __init__(
            self,
            latent_dim,
            n_comp,
            cm_type='diagonal'
        ):
        super().__init__()

        self.dim = latent_dim
        self.n_comp = n_comp
        self.cm_type=cm_type

        self.means_prior = {
            'dist': SoftballPrior(
                latent_dim=self.dim,
                radius=5,
                sharpness=5
            )
        }

        ## BEST PERFORMANCE (-5.0, 1.0)
        self.log_var_prior = {
            'dist': GaussianPrior(
                latent_dim=self.dim,
                # mean=-2 * math.log(10),
                # mean=-4.5,
                mean=-5.0,
                stddev=0.5
            )
        }

        ## BEST PERFORMANCE (0.1)
        # self.logbeta.fill_(-2 * math.log(self.sd_init[0]))
        self.log_var  = nn.Parameter(
                            torch.full(size=(self.n_comp, self.dim),
                                    #    fill_value=(0.2 * self.means_prior['dist'].radius * (self.n_comp ** -1))),
                                       fill_value=(0.2 * 0.125 * (self.n_comp ** -1))),
                                    #    fill_value=(math.log(2.0))),
                                    #    fill_value=(0.125)),
                                    #    fill_value=(0.1)),
                                    #    fill_value=(2.0)),
                                    #    fill_value=(math.log(2.0) / self.n_comp)),
                            requires_grad=True
                        )
        
        self.means    = nn.Parameter(
                           self.means_prior['dist'].sample(n_sample=self.n_comp),
                           requires_grad=True
                        )
        
        self.weights  = nn.Parameter(
                            torch.ones(self.n_comp),
                            requires_grad=True
                        )


    def get_log_prob_comp(self, x):
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
        return torch.softmax(self.weights, dim=-1)


    def get_prior_log_prob(self):
        p = 0.0

        ## Assume weights prior is Dirichlet (add alpha, constant)
        alpha = 11
        p = math.lgamma(self.n_comp * alpha) - \
            self.n_comp * math.lgamma(alpha)
        
        if alpha != 1:
            p += (alpha - 1.0) * (self.get_mixture_probs().log().sum())
        

        ## Assume means prior is Softball
        p += self.means_prior['dist'].log_prob(self.means).sum()

        ## Assume logvar prior is Gaussian
        p += self.log_var_prior['dist'].log_prob(self.log_var).sum()

        return p

    # Negative Log Probability
    def forward(self, x):
        y = self.get_log_prob_comp(x)

        y = torch.logsumexp(y, dim=-1)

        y = y + self.get_prior_log_prob()

        return y
    
    def get_mixture_probs(self):
        return torch.softmax(self.weights, dim=-1)
    

    def sample_new_points(self, n_samples):
        with torch.no_grad():
            sampled_points = torch.repeat_interleave(self.means.clone().detach().unsqueeze(0),
                                                     n_samples,
                                                     dim=0)
        
        return sampled_points.view(n_samples * self.n_comp, self.dim)



class RepresentationLayer(nn.Module):
    def __init__(
        self,
        values=None
    ):
        super().__init__()

        # If values were not provided
        if values is None:
            ## TODO: TO FIX (Add yaml file for parameters)
            # self.dim = latent_dim
            # self.n_sample = n_sample
            # self.z = nn.Parameter(
            #     torch.normal(0,
            #                  1,
            #                  size=(self.n_sample, self.dim),
            #                  requires_grad=True)
            # )
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
        if idx is None:
            return self.z
        else:
            return self.z[idx]