import torch
import torch.nn as nn

## TODO: To figure out final distribution for metabolite abundance
class LogNormalLayer(nn.Module):
    def __init__(
        self,
        output_dim,
        output_prediction_type="mean",
        output_activation_type="leakyrelu",
        reduction="none"
    ):
        super().__init__()

        # self.mean = nn.Parameter(
        #     torch.full(fill_value=0.0, size=(1, output_dim)),
        #     requires_grad=True
        # )

        self.std_dev = nn.Parameter(
            torch.full(fill_value=1.0, size=(1, output_dim)),
            requires_grad=True
        )

        self.output_prediction_type = output_prediction_type
        self.output_activation_type = output_activation_type
        self.reduction = reduction

        ## Assume activation type is leakyrelu
        self.activation_layer = nn.LeakyReLU()

    
    def forward(self, x):
        return self.activation_layer(x)
    

    def loss(x, y):
        loss = 1



class HurdleGammaLayer(nn.Module):
    def __init__(
        self,
        output_dim,
        output_prediction_type="mean",
        output_activation_type="softplus",
    ):
        super().__init__()
        
        self.alpha_shape = nn.Parameter(
            torch.full(fill_value=1.0, size=(1, output_dim)),
            requires_grad=True
        )

        self.lambda_rate = nn.Parameter(
            torch.full(fill_value=1.0, size=(1, output_dim)),
            requires_grad=True
        )

        self.pi = nn.Parameter(
            torch.full(fill_value=0.5, size=(1, output_dim)),
            requires_grad=True
        )

        self.output_prediction_type = output_prediction_type
        self.output_activation_type = output_activation_type

        ## Assume activation type is leakyrelu
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
        x_zm = (x == 0)
        x_nz = not(x_zm)
        eps = torch.full(fill_value=1.0e-10, size=x.shape)
        loss_vec = torch.zeros_like(x)

        nll_zr = torch.log(self.pi)
        nll_nz = torch.log((1 - self.pi)) + self.logGammaDensity(x, y) - torch.log(1 - self.logGammaDensity(eps, y))

        loss_vec[x_zm] = nll_zr[x_zm]
        loss_vec[x_nz] = nll_nz[x_nz]

        return loss_vec
        # return -self.logGammaDensity(x, y)
