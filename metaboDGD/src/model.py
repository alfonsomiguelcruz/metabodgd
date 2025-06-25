import torch
from torch.utils.data import DataLoader
from metaboDGD.src.decoder import Decoder
from metaboDGD.src.latent  import GaussianMixtureModel, RepresentationLayer
from metaboDGD.src.dataset import MetaboliteDataset

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
    

    def select_best_rep(self, data_loader, rep_layer):
        n_samples = len(data_loader.dataset)

        n_comp = self.gmm.n_comp

        dim = self.gmm.dim
        
        n_metabolites = self.dec.nn[-1].out_features

        best_reps = torch.empty(size=(n_samples, dim))

        for x, i in data_loader:
            n_batch_size = len(i)

            z_all = rep_layer()

            z_3d = z_all.view(
                n_samples,
                n_comp,
                dim
            )[i]

            z = z_3d.view(
                n_batch_size * n_comp,
                dim
            )

            dec_out = self.dec(z)

            # n_samples_in_batch , 1 , n_comp, n_genes
            # n_samples_in_batch , 1 , 1 , 1

            dec_fin = dec_out.view(n_batch_size, n_comp, n_metabolites)

            recon_loss = self.dec.normal_layer.loss(x,dec_fin)

            recon_loss_sum = recon_loss.sum(-1).clone()

            ## Reshape recon_loss_sum to (n_batch_size * n_comp)

            gmm_loss = self.gmm(z).clone()

            total_loss = recon_loss_sum_reshaped + gmm_loss

            ## Reshape total_loss to (n_batch_size * n_comp)

            ## best_rep_per_sample = torch.argmin(total_loss_reshaped, dim=1).squeeze(-1)

            ## Reshape z to (n_batch_size, n_comp, dim)[range(n_batch_size), best_rep_per_sample]

            best_reps[i] = rep

        return best_reps



    def get_representations(self, df, df_lbls, n_samples):
        ds = MetaboliteDataset(
            np_mat = df.to_numpy(),
            cohort_labels=df_lbls
        )

        data_loader = DataLoader(
            dataset=ds,
            batch_size=16,
            shuffle=False
        )

        ## Assume for now one representation per component
        rep_init = self.gmm.sample_new_points(n_samples)

        ## TODO: Fix representation layer to not ask for n_sample and latent_dim
        rep_layer_init = RepresentationLayer(values=rep_init)
        
        return data_loader, rep_layer_init

        # 'n_samples' *  'n_comp' * 'n_rep_per_comp' , 'dim'
        # 'n_samples' , 'n_rep_per_comp' , 'n_comp' , 'dim'
        # 'n_samples_in_batch' * 'n_comp' * 'n_rep_per_comp' , 'dim'
        

        """
        STEPS
        1. DataLoader - DONE
        2. Sample New Points from GMM Means - DONE
        3. Initialize RepresentationLayer from [2] - DONE
        4. Select the best representation from GMM Forward
        5. Initialize RepresentationLayer from [4]
        6. Initialize Optimizer (Adam)
        7. Optimize Representation
            for e in epoch:
                optimizer.zero_grad()
                for x, i in data_loader:
                    z = rep_layer_from_[5]
                    y  = dec(z)
                    recon_loss
                    gmm_loss
                    total_loss
                    total_loss.backward()
                optimizer.step()
        8. optimizer.zero_grad()
        """
        pass
