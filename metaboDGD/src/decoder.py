import math
import torch
import torch.nn as nn
import torch.distributions as D
import torch.special

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
        
        print("_______________________________")
        print(f"MIN A: {torch.min(self.alpha_shape)}")
        print(f"MIN X: {torch.min(x, axis=1)[0]}")
        print(f"MIN Y: {torch.min(y, axis=1)[0]}")
        print("_______________________________")

        p = p + self.alpha_shape * torch.log(y)
        p = p - self.alpha_shape * torch.log(self.alpha_shape)
        p = p + torch.log(x)
        p = p - self.alpha_shape * torch.log(x)
        p = p + (self.alpha_shape * x / y)
        p = p + torch.lgamma(self.alpha_shape)

        return p


    def loss(self, x, y):
        x_zm = (x == 0)
        x_nz = ~(x_zm)
        eps = torch.full(fill_value=1e-5, size=x.shape)
        loss_vec = torch.zeros_like(x)
      
        # gamma = D.Gamma(self.alpha_shape, self.alpha_shape / y)
        nll_zr = torch.log((1-self.pi)).repeat(x.shape[0], 1)
        # nll_nz = torch.log((1 - self.pi)) + gamma.log_prob(x) - torch.log(1 - torch.exp(gamma.log_prob(eps)))
        # nll_nz = torch.log(self.pi) + gamma.log_prob(x) - torch.log(1 - torch.exp(gamma.log_prob(eps)))

        # print(f"PI: {torch.any(torch.isnan(self.pi))}")
        # print(f"X :{torch.any(torch.isnan(x))}")
        # print(f"Y :{torch.any(torch.isnan(y))}")
        # print(f"EPS:{torch.any(torch.isnan(eps))}")
        # print(f"PI TERM: {torch.any(torch.isnan(torch.log((self.pi))))}")
        # print(f"PDF TERM: {torch.any(torch.isnan(self.logGammaDensity(x + 1e-5, y+1e-5)))}")
        print(f"ZER TERM: {torch.exp(self.logGammaDensity(eps, y+1e-5))}")
        
        print(f"LOSS_MIN X: {torch.min(x, axis=1)[0]}")
        print(f"LOSS_MIN Y: {torch.min(y, axis=1)[0]}")
        print(f"LOSS_MIN PI: {torch.min(self.pi)}")
        # print(self.logGammaDensity(x + 1e-5, y))
        nll_nz = torch.log(self.pi) + self.logGammaDensity(x + 1e-5, y + 1e-5) #- torch.log(1 - torch.special.gammainc(self.alpha_shape, self.alpha_shape * x / y))

        print(f"NLL_NZ: {nll_nz}")
        print(f"NANS_ZR: {torch.any(torch.isnan(nll_zr), dim=1, keepdim=True)}")
        print(f"NANS_NZ: {torch.any(torch.isnan(nll_nz), dim=1, keepdim=True)}")

        loss_vec[x_zm] = nll_zr[x_zm]
        loss_vec[x_nz] = nll_nz[x_nz]
        return loss_vec
    

#### FOR EXPERIMENTATION ONLY
class HurdleNormalLayer (nn.Module):
    def __init__(
        self,
        output_dim,
        output_prediction_type="mean",
        output_activation_type="leakyrelu",
    ):
        super().__init__()
        
        self.std = nn.Parameter(
            torch.full(fill_value=1.0, size=(1, output_dim), dtype=torch.float64),
            requires_grad=True
        )

        # self.lambda_rate = nn.Parameter(
        #     torch.full(fill_value=0.1, size=(1, output_dim), dtype=torch.float64),
        #     requires_grad=True
        # )

        self.pi = nn.Parameter(
            torch.full(fill_value=0.5, size=(1, output_dim), dtype=torch.float64),
            requires_grad=True
        )

        self.output_prediction_type = output_prediction_type
        self.output_activation_type = output_activation_type

        ## Assume activation type is softplus
        self.activation_layer = nn.LeakyReLU()

    
    def forward(self, x):
        return self.activation_layer(x)
    

    def logGaussianDensity(self, x, y):
        p = 0.0

        print("_______________________________")
        print(f"MIN V: {torch.min(self.var)}")
        print(f"MIN X: {torch.min(x, axis=1)[0]}")
        print(f"MIN Y: {torch.min(y, axis=1)[0]}")
        print("_______________________________")

        p = p + ((x-y)**2).div(self.var)
        p = p + torch.log(torch.Tensor([2 * torch.pi]))
        p = p + torch.log(self.var)
        p = p * 0.5

        print(f"p: {p}")
        return p


    def loss(self, x, y):
        x_zm = (x == 0)
        x_nz = ~(x_zm)
        eps = torch.full(fill_value=1e-5, size=x.shape)
        loss_vec = torch.zeros_like(x)
        
        normal = D.Normal(loc=y+1e-5, scale=self.std)
        nll_zr = torch.log((1 - self.pi)).repeat(x.shape[0], 1)
        nll_nz = torch.log(self.pi) + normal.log_prob(x+1e-5) - torch.log(1 - normal.cdf(eps))
        # nll_nz = torch.log((self.pi)) + self.logGaussianDensity(x + 1e-5, y + 1e-5) - torch.log(1 - torch.exp(self.logGaussianDensity(eps, y)))
        
        # print(f"PI: {torch.any(torch.isnan(self.pi))}")
        # print(f"X :{torch.any(torch.isnan(x))}")
        # print(f"Y :{torch.any(torch.isnan(y))}")
        # print(f"EPS:{torch.any(torch.isnan(eps))}")
        # print(f"PI TERM: {torch.any(torch.isnan(torch.log((1 - self.pi))))}")
        # print(f"PDF TERM: {torch.any(torch.isnan(self.logGammaDensity(x + 1e-5, y)))}")
        # print(f"ZER TERM: {torch.any(torch.isnan(torch.log(1 - torch.exp(self.logGammaDensity(eps, y)))))}")
        # print(f"MIN X: {torch.min(x, axis=1)}")
        # print(f"MIN Y: {torch.min(y, axis=1)}")

        # print(self.logGammaDensity(x + 1e-5, y))
        # print(f"NLL_NONZERO: {nll_nz}")
        # print(f"NANS: {torch.any(torch.isnan(nll_nz), dim=1, keepdim=True)}")
        print(f"NANS_ZR: {torch.any(torch.isnan(nll_zr), dim=1, keepdim=True)}")
        print(f"NANS_NZ: {torch.any(torch.isnan(nll_nz), dim=1, keepdim=True)}")

        loss_vec[x_zm] = nll_zr[x_zm]
        loss_vec[x_nz] = nll_nz[x_nz]
        return loss_vec




#### FOR EXPERIMENTATION ONLY
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

        self.pi = nn.Parameter(
            torch.full(fill_value=0.5, size=(1, output_dim), dtype=torch.float64),
            requires_grad=True
        )

        self.output_prediction_type = output_prediction_type
        self.output_activation_type = output_activation_type

        ## Assume activation type is softplus
        self.activation_layer = nn.LeakyReLU()

    
    def forward(self, x):
        return self.activation_layer(x)


    def loss(self, x, y):
        # x_zm = (x == 0)
        # x_nz = ~(x_zm)
        # eps = torch.full(fill_value=1e-5, size=x.shape)
        # loss_vec = torch.zeros_like(x)
        
        # lognormal = D.LogNormal(loc=y, scale=self.std)
        # nll_zr = torch.log((1 - self.pi)).repeat(x.shape[0], 1)
        # nll_nz = torch.log(self.pi) + lognormal.log_prob(x+1e-5) - torch.log(1 - lognormal.cdf(eps))

        # if torch.any(torch.isnan(nll_zr)) or torch.any(torch.isnan(nll_nz)):
        #     print("NAN FOUND!")
            
        # loss_vec[x_zm] = nll_zr[x_zm]
        # loss_vec[x_nz] = nll_nz[x_nz]
        # return -loss_vec

        eps = torch.full(fill_value=1e-5, size=x.shape)

        # if torch.any(torch.isnan(y)) or torch.any(torch.isnan(self.std)):
        #     print("NAN FOUND! (MEAN, STD)")

        normal = D.Normal(loc=y, scale=self.std)
        # nll_zr = torch.log((1 - self.pi)).repeat(x.shape[0], 1)
        nll_zr = torch.log((1 - self.pi)).expand_as(x)
        nll_nz = torch.log(self.pi) + normal.log_prob(x+1e-5) - torch.log(1 - normal.cdf(eps))
        # print('-------------------')
        # print(f"nll_zr: {nll_zr.shape}")
        # print(f"nll_nz: {nll_nz.shape}")
        # print(f"x: {x.shape}")
        # print(f"y: {y.shape}")
        # print('-------------------')
        
        # if torch.any(torch.isnan(nll_zr)) or torch.any(torch.isnan(nll_nz)):
        #     print("NAN FOUND! (NLL_ZR, NLL_NZ)")

        return -torch.where(x > 0, nll_nz, nll_zr)


    



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
                self.nn.append(nn.LeakyReLU(True))
            elif i == n_hidden_layers - 1:
                self.nn.append(nn.Linear(hidden_layer_dim[-1], output_layer_dim))
            else:
                self.nn.append(nn.Linear(hidden_layer_dim[i-1], hidden_layer_dim[i]))
                self.nn.append(nn.LeakyReLU(True))

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