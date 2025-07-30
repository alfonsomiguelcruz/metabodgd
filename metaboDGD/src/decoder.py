import math
import torch
import torch.nn as nn
import torch.distributions as D

class HurdleLogNormalLayer (nn.Module):
    def __init__(
        self,
        output_dim,
        output_prediction_type="mean",
        output_activation_type="leakyrelu",
    ):
        super().__init__()
        
        self.std = nn.Parameter(
            torch.full(fill_value=3.0, size=(1, output_dim), dtype=torch.float64),
            requires_grad=True
        )

        self.output_prediction_type = output_prediction_type
        self.output_activation_type = output_activation_type

        ## Assume activation type is softplus
        self.activation_layer = nn.LeakyReLU()

    
    def forward(self, x):
        return self.activation_layer(x)


    def loss(self, x, y):
        eps = torch.full(fill_value=1e-5, size=x.shape)

        normal = D.Normal(loc=y, scale=self.std)
        
        recon_loss = -normal.log_prob(x+eps)

        return recon_loss


class Decoder(nn.Module):
    def __init__(
        self,
        latent_layer_dim,
        output_layer_dim,
        hidden_layer_dim=[100,100,100],
        output_prediction_type="mean",
        output_activation_type="softplus"
    ):
        super().__init__()

        self.nn = nn.ModuleList()

        n_hidden_layers = len(hidden_layer_dim) + 1
        for i in range(n_hidden_layers):
            if i == 0:
                self.nn.append(nn.Linear(latent_layer_dim, hidden_layer_dim[i]))
                self.nn.append(nn.PReLU())
            elif i == n_hidden_layers - 1:
                self.nn.append(nn.Linear(hidden_layer_dim[-1], output_layer_dim))
            else:
                self.nn.append(nn.Linear(hidden_layer_dim[i-1], hidden_layer_dim[i]))
                self.nn.append(nn.PReLU())

        # TODO Figure out loss function distribution for metabolite abundance
        self.normal_layer = HurdleLogNormalLayer(
            output_dim=output_layer_dim,
            output_prediction_type=output_prediction_type,
            output_activation_type=output_activation_type
        )
    
    
    def forward(self, z):
        for i in range(len(self.nn)):
            z = self.nn[i](z)

        return self.normal_layer(z)


    @classmethod
    def load(cls):
        pass