import math
import torch
import torch.nn as nn


class HurdleGammaLayer(nn.Module):
    def __init__(
        self,
        output_dim,
        output_prediction_type="mean",
        output_activation_type="softplus",
    ):
        super().__init__()
        
        self.alpha_shape = nn.Parameter(
            torch.full(fill_value=1.0, size=(1, output_dim), dtype=torch.float64),
            requires_grad=True
        )

        self.lambda_rate = nn.Parameter(
            torch.full(fill_value=1.0, size=(1, output_dim), dtype=torch.float64),
            requires_grad=True
        )

        self.pi = nn.Parameter(
            torch.full(fill_value=0.5, size=(1, output_dim), dtype=torch.float64),
            requires_grad=True
        )

        self.output_prediction_type = output_prediction_type
        self.output_activation_type = output_activation_type

        ## Assume activation type is softplus
        self.activation_layer = nn.Softplus()

    
    def forward(self, x):
        return self.activation_layer(x)
    

    def logGammaDensity(self, x, y):
        p = 0.0
        p += self.alpha_shape * torch.log(y)
        p -= self.alpha_shape * torch.log(self.alpha_shape)
        p += torch.log(x)
        p -= self.alpha_shape * torch.log(x)
        p += (self.alpha_shape * x / y)
        p += torch.lgamma(self.alpha_shape)

        return p


    def loss(self, x, y):
        x_zm = (x == 1e-10)
        x_nz = ~(x_zm)
        eps = torch.full(fill_value=1.0e-10, size=x.shape)
        loss_vec = torch.zeros_like(x)
      
        # print(loss_vec.dtype)
        nll_zr = torch.log(self.pi).repeat(x.shape[0], 1)
        nll_nz = torch.log((1 - self.pi)) + self.logGammaDensity(x, y) - torch.log(1 - self.logGammaDensity(eps, y))
        # print(nll_zr.dtype)
        loss_vec[x_zm] = nll_zr[x_zm]
        loss_vec[x_nz] = nll_nz[x_nz]

        return loss_vec
        # return -self.logGammaDensity(x, y)



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
                self.nn.append(nn.ReLU(True))
            elif i == n_hidden_layers - 1:
                self.nn.append(nn.Linear(hidden_layer_dim[-1], output_layer_dim))
            else:
                self.nn.append(nn.Linear(hidden_layer_dim[i-1], hidden_layer_dim[i]))
                self.nn.append(nn.ReLU(True))

        # TODO Figure out loss function distribution for metabolite abundance
        self.gamma_layer = HurdleGammaLayer(output_dim=output_layer_dim)
    
    
    def forward(self, z):
        for i in range(len(self.nn)):
            z = self.nn[i](z)

        return self.gamma_layer(z)

        

    @classmethod
    def load(cls):
        pass