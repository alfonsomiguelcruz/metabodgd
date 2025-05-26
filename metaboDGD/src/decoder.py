import torch.nn as nn

## TODO: To figure out final distribution for metabolite abundance
class LogNormalLayer(nn.Module):
    def __init__(self):
        super().__init__()


class Decoder(nn.Module):
    def __init__(
        self,
        latent_layer_dim,
        output_layer_dim,
        hidden_layer_dim=[100,100,100],
        r_init=2,
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
        self.output_layer = None
    
    
    def forward(self, z):
        for i in range(len(self.nn)):
            z = self.nn[i](z)

        return z

        

    @classmethod
    def load(cls):
        pass