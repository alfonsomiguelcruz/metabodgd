import torch
import pickle
import yaml

from metaboDGD.util import data, train
from metaboDGD.src import model
from metaboDGD.src.dataset import MetaboliteDataset
from metaboDGD.src.latent import RepresentationLayer

#####     Directories and Initialize Data     #####
dir = 'outputs/'
df_normal_fname = 'CombinedDataset_CAMP_Normal.csv'
df_tumor_fname = 'CombinedDataset_CAMP_Tumor.csv'
cohorts_normal_fname = 'cohorts_Normal.pkl'
cohorts_tumor_fname = 'cohorts_Tumor.pkl'

np_normal_lbls, np_raw_normal_lbls, np_normal_log, cohorts_normal = \
    data.load_dataframe_and_cohort(dir, df_normal_fname, cohorts_normal_fname)

np_tumor_lbls, np_raw_tumor_lbls, np_tumor_log, cohorts_tumor = \
    data.load_dataframe_and_cohort(dir, df_tumor_fname, cohorts_tumor_fname)


#####     DataLoader and Training Model    #####
train_all_loader = \
    data.create_dataloaders(np_train_abun=np_normal_log,
                            np_train_lbls=np_raw_normal_lbls,
                            batch_size=256)


config_model = yaml.safe_load(open('model.yaml', 'r'))
config_train = yaml.safe_load(open('train.yaml', 'r'))

dgd_model = model.MetaboDGD(**config_model)
dgd_model, train_rep, history, cm = \
    train.train_dgd(
        dgd_model=dgd_model,
        train_loader=train_all_loader,
        **config_train
    )


#####     Results     #####
results = {
    'np_normal_lbls': np_normal_lbls,
    'np_raw_normal_lbls': np_raw_normal_lbls,
    'np_normal_log': np_normal_log,
    'cohorts_normal': cohorts_normal,
    'np_tumor_lbls': np_tumor_lbls,
    'np_raw_tumor_lbls': np_raw_tumor_lbls,
    'np_tumor_log': np_tumor_log,
    'cohorts_tumor': cohorts_tumor,
    'history': history,
    'cm': cm,
}

f = open(dir + 'results.pkl', 'wb')
pickle.dump(results, f)
f.close()


#####     Get Normal Representations from Disease Samples     #####
dgd_final = model.MetaboDGD(**config_model)
dgd_final.dec.load_state_dict(torch.load('torch_outputs/torch_outputs_dec.pt'))
dgd_final.gmm.load_state_dict(torch.load('torch_outputs/torch_outputs_gmm.pt'))

train_rep_final = RepresentationLayer(values=torch.zeros(size=(np_normal_log.shape[0], dgd_final.gmm.dim)))
train_rep_final.load_state_dict(torch.load('torch_outputs/torch_outputs_train_rep.pt'))

tumor_ds = MetaboliteDataset(
    np_mat=np_tumor_log,
    cohort_labels=np_raw_tumor_lbls,
)

config_opt = yaml.safe_load(open('opt.yaml', 'r'))
tumor_rep_final, dec_out_final = dgd_model.get_representations(tumor_ds, config_opt)


tumor_results = {
    'tumor_rep_final': tumor_rep_final.detach().numpy(),
    'dec_out_final': dec_out_final.detach().numpy()
}

f = open(dir + 'tumor_results.pkl', 'wb')
pickle.dump(tumor_results, f)
f.close()