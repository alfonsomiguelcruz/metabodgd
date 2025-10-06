import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

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
        dec_hidden_layers_dim=[500, 1500],
        dec_output_prediction_type='mean',
        dec_output_activation_type='softplus',
        n_comp=8,
        cm_type='diagonal',
        softball_radius=3,
        softball_sharpness=5,
        gaussian_mean=-6.0,
        gaussian_stddev=0.01,
        dirichlet_alpha=0.5,
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
            cm_type=cm_type,
            softball_radius=softball_radius,
            softball_sharpness=softball_sharpness,
            gaussian_mean=gaussian_mean,
            gaussian_stddev=gaussian_stddev,
            dirichlet_alpha=dirichlet_alpha,
        )
    

    def select_best_rep(dgd, data_loader, rep_layer):
        n_samples = len(data_loader.dataset)

        n_comp = dgd.gmm.n_comp

        dim = dgd.gmm.dim
        
        n_metabolites = dgd.dec.nn[-1].out_features

        best_reps = torch.empty(size=(n_samples, dim))

        for x, i in data_loader:
            n_batch_size = len(i)

            # n_samples * n_comp * n_rep_per_comp , dim
            z_all = rep_layer()

            # n_samples , n_rep_per_comp , n_comp, dim
            z_3d = z_all.view(
                n_samples,
                n_comp,
                dim
            )[i]

            # n_samples_in_batch * n_comp * n_rep_per_comp , dim
            z = z_3d.view(
                n_batch_size * n_comp,
                dim
            )

            # n_samples_in_batch * n_comp * n_rep_per_comp , n_genes
            dec_out = dgd.dec(z)

            # n_samples_in_batch , 1 , n_comp, n_genes
            obs_counts = x.unsqueeze(1).expand(-1, n_comp ,-1)

            # n_samples_in_batch , 1 , 1 , 1
            
            # n_samples_in_batch , n_rep_per_comp , n_comp, n_genes
            # n_samples_in_batch, n_comp, n_metabolites
            pred_means = dec_out.view(n_batch_size, n_comp, n_metabolites)

            # print(obs_counts.shape)
            # print(pred_means.shape)
            recon_loss = dgd.dec.normal_layer.loss(obs_counts, pred_means)
            recon_loss_sum = recon_loss.sum(-1).clone()

            recon_loss_sum_reshaped = recon_loss_sum.view(n_batch_size * n_comp)

            gmm_loss = dgd.gmm(z).clone()

            total_loss = recon_loss_sum_reshaped + gmm_loss
            total_loss_reshaped = total_loss.view(n_batch_size, n_comp)
            
            best_rep_per_sample = torch.argmin(total_loss_reshaped, dim=1).squeeze(-1)

            rep = z.view(n_batch_size, n_comp, dim)[range(n_batch_size), best_rep_per_sample]

            best_reps[i] = rep
            ## Reshape z to (n_batch_size, n_comp, dim)[range(n_batch_size), best_rep_per_sample]

            # best_reps[i] = rep

        return best_reps



    def get_representations(self, np, np_lbls, n_samples):
        ds = MetaboliteDataset(
            np_mat = np,
            cohort_labels=np_lbls
        )

        data_loader = DataLoader(
            dataset=ds,
            batch_size=32,
            shuffle=False
        )

        ## Assume for now one representation per component
        rep_init = self.gmm.sample_new_points(n_samples)

        ## TODO: Fix representation layer to not ask for n_sample and latent_dim
        rep_layer_init = RepresentationLayer(values=rep_init)

        rep_best = self.select_best_rep(data_loader=data_loader, rep_layer=rep_layer_init)

        rep_layer_best = RepresentationLayer(values=rep_best)

        optimizer = Adam(
            params=rep_layer_best.parameters(),
            lr=1e-3, # From lr_rep in training
            weight_decay=0.0,
            betas=[0.5, 0.7]
        )

        """
        rep, dec_out, time = 
            self._optimize_rep(
                data_loader = data_loader, [DONE]
                rep_layer = rep_layer_best,[DONE]
                optimizer = optimizer,     [DONE]
                n_comp = 1,                [1]
                n_rep_per_comp = 1,        [EXEMPT]
                epochs = epochs,           [100]
                opt_num = 1)               [?]
        """
        n_metabolites = self.dec.nn[-1].out_features
        dim_op = self.gmm.dim
        n_comp_op = 1
        n_samples_op = len(data_loader.dataset)
        n_epochs = 10
        for e in range(0, n_epochs):
            optimizer.zero_grad()
            for x, i in data_loader:
                n_batch_size = len(i)
                z_all = rep_layer_best()
                # print(z_all.shape)

                z_3d = z_all.view(n_samples_op, n_comp_op, dim_op)[i]

                z = z_3d.view(n_batch_size * n_comp_op, dim_op)

                dec_out = self.dec(z=z)

                obs_counts = x.unsqueeze(1).expand(-1, n_comp_op ,-1)

                pred_means = dec_out.view(n_batch_size, n_comp_op, n_metabolites)

                recon_loss = self.dec.normal_layer.loss(obs_counts, pred_means)

                recon_loss_sum = recon_loss.sum().clone()

                gmm_loss = self.gmm(z)

                gmm_loss_sum = gmm_loss.sum().clone()

                total_loss = recon_loss_sum + gmm_loss_sum

                total_loss.backward()
            optimizer.step()
            if e + 1 == n_epochs:
                rep_final = rep_layer_best()
                dec_out_final = self.dec(rep_final)
                return rep_final, dec_out_final
        # 'n_samples' *  'n_comp' * 'n_rep_per_comp' , 'dim'
        # 'n_samples' , 'n_rep_per_comp' , 'n_comp' , 'dim'
        # 'n_samples_in_batch' * 'n_comp' * 'n_rep_per_comp' , 'dim'

        optimizer.zero_grad()