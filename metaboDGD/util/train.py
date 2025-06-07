import os
import torch
import pandas as pd
import numpy as np
from torch.optim import Adam
from metaboDGD.src.latent import RepresentationLayer

def train_dgd(
    dgd_model,
    train_loader,
    validation_loader,
    n_epochs=500,
    export_dir='./',
    export_name='metaboDGD',
    lr_schedule_epochs=[0,300],
    lr_schedule=[[1e-4,1e-3,1e-2],[1e-4,1e-2,1e-2]],
    optim_betas=[0.5,0.7],
    wd=1e-4,
    acc_save_threshold=0.5
):
    # Saving the model
    if export_name is not None:
        if not os.path.exists(export_dir+export_name):
            os.makedirs(export_dir+export_name)

    # Get information about the data
    nsample_train = len(train_loader.dataset)
    nsample_val   = len(validation_loader.dataset)

    out_dim    = train_loader.dataset.n_metabolites
    latent_dim = dgd_model.gmm.dim

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dgd_model.dec = dgd_model.dec.to(device)
    dgd_model.gmm = dgd_model.gmm.to(device)


    # Setting up Representations and Optimizers
    if lr_schedule_epochs is None:
        lr_dec, lr_rep, lr_gmm = lr_schedule
    else:
        lr_dec, lr_rep, lr_gmm = lr_schedule[0]
    
    train_rep = RepresentationLayer(
        latent_dim=latent_dim,
        n_sample=nsample_train,
        values=torch.zeros(size=(nsample_train, latent_dim))
    ).to(device)

    val_rep = RepresentationLayer(
        latent_dim=latent_dim,
        n_sample=nsample_val,
        values=torch.zeros(size=(nsample_val, latent_dim))
    ).to(device)

    train_rep_optimizer = Adam(
        params=train_rep.parameters(),
        lr=lr_rep,
        weight_decay=wd,
        betas=(optim_betas[0], optim_betas[1])
    )

    val_rep_optimizer = Adam(
        params=val_rep.parameters(),
        lr=lr_rep,
        weight_decay=wd,
        betas=(optim_betas[0], optim_betas[1])
    )

    gmm_optimizer = Adam(
        params=dgd_model.gmm.parameters(),
        lr=lr_gmm,
        weight_decay=wd,
        betas=(optim_betas[0], optim_betas[1])
    )

    dec_optimizer = Adam(
        params=dgd_model.dec.parameters(),
        lr=lr_dec,
        weight_decay=wd,
        betas=(optim_betas[0], optim_betas[1])
    )

    # Tracking of Losses and Other Metrics
    train_avg       = []
    recon_avg       = []
    dist_avg        = []

    val_avg         = []
    recon_val_avg   = []
    dist_val_avg    = []

    cluster_accuracies = []
    best_gmm_cluster = 0

    for e in range(n_epochs):
        if lr_schedule_epochs is not None:
            if e in lr_schedule_epochs:
                lr_idx = [x for x in range(len(lr_schedule_epochs)) \
                          if lr_schedule_epochs[x] == e][0]
                
                lr_dec = lr_schedule[lr_idx][0]
                dec_optimizer = Adam(
                    params=dgd_model.dec.parameters(),
                    lr=lr_dec,
                    weight_decay=wd,
                    betas=(optim_betas[0], optim_betas[1])
                )

                lr_rep = lr_schedule[lr_idx][1]

                lr_gmm = lr_schedule[lr_idx][2]
                gmm_optimizer = Adam(
                    params=dgd_model.gmm.parameters(),
                    lr=lr_gmm,
                    weight_decay=wd,
                    betas=(optim_betas[0], optim_betas[1])
                )
                

        train_avg.append(0)
        recon_avg.append(0)
        dist_avg.append(0)

        val_avg.append(0)
        recon_val_avg.append(0)
        dist_val_avg.append(0)

        ## Training Run
        dgd_model.dec.train()
        train_rep_optimizer.zero_grad()
        for x, i in train_loader:
            gmm_optimizer.zero_grad()
            dec_optimizer.zero_grad()

            x = x.to(device)
            z = train_rep(i)     
            y = dgd_model.dec(z)

            recon_loss = dgd_model.dec.normal_layer.loss(x, y).sum()
            dist_loss  = -dgd_model.gmm(z).sum()
            loss = recon_loss.clone() + dist_loss.clone()

            loss.backward()
            gmm_optimizer.step()
            dec_optimizer.step()

            train_avg[-1] += loss.item() / (nsample_train * out_dim)
            recon_avg[-1] += recon_loss.item() / (nsample_train * out_dim)
            dist_avg[-1]  += dist_loss.item() / (nsample_train * latent_dim)
            
        train_rep_optimizer.step()

        ## TODO: cluster_accuracies.append()

        ## Validation Run
        dgd_model.dec.eval()
        val_rep_optimizer.zero_grad()
        for x, i in validation_loader:
            x = x.to(device)
            z = val_rep(i)            
            y = dgd_model.dec(z)

            recon_loss = dgd_model.dec.normal_layer.loss(x, y).sum()
            dist_loss  = -dgd_model.gmm(z).sum()
            loss = recon_loss.clone() + dist_loss.clone()
            
            loss.backward()

            val_avg[-1] += loss.item() / (nsample_val * out_dim)
            recon_val_avg[-1] += recon_loss.item() / (nsample_val * out_dim)
            dist_val_avg[-1]  += dist_loss.item() / (nsample_val * latent_dim)
        
        val_rep_optimizer.step()

    history = pd.DataFrame({
        'train_loss': train_avg,
        'val_loss': val_avg,
        'train_recon_loss': recon_avg,
        'val_recon_loss': recon_val_avg,
        'train_dist_loss': dist_avg,
        'val_dist_loss': dist_val_avg,
        'epoch': np.arange(1, n_epochs+1),
    })

    return dgd_model, train_rep, val_rep, history