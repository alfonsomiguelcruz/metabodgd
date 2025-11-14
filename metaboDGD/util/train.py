import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

from metaboDGD.src.latent import RepresentationLayer


def get_cluster_accuracy(rep, gmm, labels):
    """
    Computes the GMM clustering quality using the
    clustering accuracy. 

    Parameters
    ----------
    rep : `RepresentationLayer`
        The representations of the training data.

    gmm : `GaussianMixtureModel`
        The GMM being trained.

    labels : `list`
        A list of strings containing the cohorts the samples
        originally belonged to.


    Returns
    -------
    cm2 : `numpy.ndarray`
        A confusion matrix between the true and predicted
        labels by the GMM.

    acc : `float`
        A float denoting the computed clustering accuracy.
    """
        
    le = LabelEncoder()
    true_labels = le.fit_transform(labels)

    clustering  = torch.exp(gmm.get_log_prob_comp(rep.z.detach()))
    pred_labels = torch.max(clustering, dim=-1).indices.cpu().detach()

    cm = confusion_matrix(true_labels, pred_labels)
    
    idxs = linear_sum_assignment(-cm + np.max(cm))
    cm2 = cm[:, idxs[1]]

    acc = np.trace(cm2) / np.sum(cm2)

    return cm2, acc


def train_dgd(
    dgd_model,
    train_loader,
    validation_loader=None,
    n_epochs=300,
    export_dir='./',
    export_name='torch_outputs',
    lr_schedule_epochs=[0,300],
    lr_schedule=[[1e-4,1e-3,1e-2],[1e-4,1e-2,1e-2]],
    optim_betas=[0.5,0.7],
    wd=1e-4,
    acc_save_threshold=0.4,
    save_here=True,
):
    # Hello!
    # Saving the model
    if export_name is not None:
        if not os.path.exists(export_dir+export_name):
            os.makedirs(export_dir+export_name)

    # Get information about the data
    nsample_train = len(train_loader.dataset)

    if validation_loader:
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
        values=torch.zeros(size=(nsample_train, latent_dim))
    ).to(device)

    if validation_loader:
        val_rep = RepresentationLayer(
            values=torch.zeros(size=(nsample_val, latent_dim))
        ).to(device)

    train_rep_optimizer = Adam(
        params=train_rep.parameters(),
        lr=lr_rep,
        weight_decay=wd,
        betas=(optim_betas[0], optim_betas[1])
    )

    if validation_loader:
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

    if validation_loader:
        val_avg         = []
        recon_val_avg   = []
        dist_val_avg    = []

    cluster_accuracies = []
    best_gmm_cluster = 0
    dir_temp = export_dir + export_name + '/' + export_name

    for e in range(n_epochs):
        # if e % 10 == 0:
        #     print(f"EPOCH: {e}")
        #     show_pca_plot(normal_lbls, cohorts, train_rep, dgd_model)
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

        if validation_loader:
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

        cm2, acc = get_cluster_accuracy(train_rep, dgd_model.gmm, train_loader.dataset.get_labels())
        cluster_accuracies.append(acc)
        best_labels = cm2

        ## Validation Run
        if validation_loader:
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

        if best_gmm_cluster < acc and acc > acc_save_threshold:
            best_gmm_cluster = acc
            best_gmm_epoch = e
            print(f'Cluster Acc: {acc}')
            
            # Save torch files if needed (not required for Hyperparameter Search)
            if save_here:
                torch.save(dgd_model.dec.state_dict(), dir_temp+'_dec.pt')
                torch.save(dgd_model.gmm.state_dict(), dir_temp+'_gmm.pt')
                torch.save(train_rep.state_dict(), dir_temp+'_train_rep.pt')
                
                if validation_loader:
                    torch.save(val_rep.state_dict(), dir_temp+'_val_rep.pt')

    if validation_loader:
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
    else:
        history = pd.DataFrame({
            'train_loss': train_avg,
            'train_recon_loss': recon_avg,
            'train_dist_loss': dist_avg,
            'cluster_acc': cluster_accuracies,
            'epoch': np.arange(1, n_epochs+1),
        })

        if save_here:
            torch.save(dgd_model.dec.state_dict(), dir_temp+'_dec.pt')
            torch.save(dgd_model.gmm.state_dict(), dir_temp+'_gmm.pt')
            torch.save(train_rep.state_dict(), dir_temp+'_train_rep.pt')
            print("Saving Model...")
        
        return dgd_model, train_rep, history, best_labels


def get_history_plot(history, inc_gmm_acc=True, with_val_plot=True):

    if inc_gmm_acc:
        fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16,4))
        plt.subplots_adjust(wspace=0.25, hspace=0.25)

        for j in range(0,4):
            ax[j].set_xlabel('Epoch')
                
            match j:
                case 0:
                    train_lbl = 'train_loss'
                    title = 'Total Loss'
                    if with_val_plot:
                        val_lbl = 'val_loss'
                case 1:
                    train_lbl = 'train_recon_loss'
                    title = 'Reconstruction Loss'
                    if with_val_plot:
                        val_lbl = 'val_recon_loss'
                case 2:
                    train_lbl = 'train_dist_loss'
                    title = 'Distribution Loss'
                    if with_val_plot:
                        val_lbl = 'val_dist_loss'
                case 3:
                    ax[j].plot(history['epoch'], history['cluster_acc'])
                    ax[j].set_ylabel('accuracy')
                    ax[j].set_title('Clustering Accuracy')


            if j != 3:
                ax[j].plot(history['epoch'], history[train_lbl], label='train')

                if with_val_plot:
                    ax[j].plot(history['epoch'], history[val_lbl], label='validation') 

                ax[j].set_ylabel('loss')                 
                ax[j].set_title(title)
                ax[j].legend()

        plt.show()

    else:
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,4))
        plt.subplots_adjust(wspace=0.20, hspace=0.25)
        
        for i,j in [(0,0), (0,1), (0,2)]:
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
                title = 'Distribution Loss'
            
            ax[j].set_xlabel('Epoch')
            ax[j].plot(history['epoch'], history[train_lbl], label='Train')
            ax[j].plot(history['epoch'], history[val_lbl], label='Validation')
            ax[j].set_ylabel('Loss')
            ax[j].set_title(title)
            ax[j].legend()


