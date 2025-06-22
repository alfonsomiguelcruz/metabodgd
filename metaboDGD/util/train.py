import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from scipy.optimize import linear_sum_assignment

from metaboDGD.src.latent import RepresentationLayer


def gmm_cluster_acc(rep, gmm, labels):
    le = LabelEncoder()
    true_labels = le.fit_transform(labels)

    clustering  = torch.exp(gmm.get_log_prob_comp(rep.z.detach()))
    pred_labels = torch.max(clustering, dim=-1).indices.cpu().detach()

    _cm  = confusion_matrix(true_labels, pred_labels)
    idxs = linear_sum_assignment(np.max(_cm) - _cm)
    cm   = _cm[:, idxs[1]]
    
    return cm, adjusted_rand_score(true_labels, pred_labels)


# def gmm_cluster_acc(rep, gmm, labels):
#     le = LabelEncoder()
#     true_labels = le.fit_transform(labels)

#     clustering  = torch.exp(gmm.get_log_prob_comp(rep.z.detach()))
#     pred_labels = torch.max(clustering, dim=-1).indices.cpu().detach()

#     cm = confusion_matrix(true_labels, pred_labels)
    
#     idxs = linear_sum_assignment(-cm + np.max(cm))
#     cm2 = cm[:, idxs[1]]
#     print(cm)
#     acc = np.trace(cm2) / np.sum(cm2)

#     return cm2, acc


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
    best_labels = None

    for e in range(n_epochs):
        if e % 20 == 0:
            print(e)
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

        cm2, acc = gmm_cluster_acc(train_rep, dgd_model.gmm, train_loader.dataset.get_labels())
        cluster_accuracies.append(acc)
        best_labels = cm2

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
        'cluster_acc': cluster_accuracies,
        'epoch': np.arange(1, n_epochs+1),
    })

    return dgd_model, train_rep, val_rep, history, best_labels


def get_history_plot(history):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,8))
    plt.subplots_adjust(wspace=0.25, hspace=0.25)

    for i,j in [(0,0), (0,1), (1,0), (1,1)]:
        if i == 1 and j == 1:
            ax[1][1].plot(history['epoch'], history['cluster_acc'])
            ax[1][1].set_xlabel('Epoch')
            ax[1][1].set_ylabel('Adjusted Rand Index')
            ax[1][1].set_title('ARI Clustering Metric')
        else:
            if i == 0 and j == 0:
                train_lbl = 'train_loss'
                val_lbl = 'val_loss'
                title = 'Total Loss'
            elif i == 0 and j == 1:
                train_lbl = 'train_recon_loss'
                val_lbl = 'val_recon_loss'
                title = 'Reconstruction Loss'
            else:
                train_lbl = 'train_dist_loss'
                val_lbl = 'val_dist_loss'
                title = 'GMM Distribution Loss'
                ax[1][0].set_xlabel('Epoch')

            ax[i][j].plot(history['epoch'], history[train_lbl], label='train')
            ax[i][j].plot(history['epoch'], history[val_lbl], label='validation')
            ax[i][j].set_ylabel('loss')
            ax[i][j].set_title(title)
            ax[i][j].legend()