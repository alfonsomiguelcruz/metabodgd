import torch
from metaboDGD.src.decoder import Decoder
from metaboDGD.src.latent  import GaussianMixtureModel

class MetaboDGD():
    # TODO:
    # Create the initialization of parameters based on bulkDGD
    # (dim, n_comp, etc.)
    def __init__ (self,
        latent_dim=20,
        output_dim=1915,
        dec_hidden_layers_dim=[475, 950, 1425],
        dec_output_prediction_type='mean',
        dec_output_activation_type='softplus',
        n_comp=8,
        cm_type='diagonal'
    ):
        super(MetaboDGD, self).__init__()

        self.dec = Decoder(
            latent_layer_dim=latent_dim,
            output_layer_dim=output_dim,
            hidden_layer_dim=dec_hidden_layers_dim,
            output_prediction_type=dec_output_prediction_type,
            output_activation_type=dec_output_activation_type
        )
        self.gmm = GaussianMixtureModel(
            latent_dim=latent_dim,
            n_comp=n_comp,
            cm_type=cm_type
        )
