import pandas as pd
import numpy  as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from metaboDGD.src.dataset import MetaboliteDataset



def get_cohort_samples(cohort_name, sample_type='Normal'):
    # Get the xls file
    xls = pd.ExcelFile(f'data/PreprocessedData_{cohort_name}.xlsx')

    # Get the dataframes for the preprocessed metabolomics data
    n = xls.parse(f"metabo_imputed_filtered_{sample_type}")

    # Get list of metabolites
    n_met_list = n["Unnamed: 0"].to_list()

    # Get list of sample IDs
    n_sample_list = n.columns.to_list()[1:]

    # Set Index to metabolite names, drop "Unnamed: 0" column
    n.set_index("Unnamed: 0", inplace=True)

    return {
        "sample_list"   : n_sample_list,
        "met_list"      : n_met_list,
        "matrix"        : n
    }

def construct_training_dataframe(met_list_all, sample_list_all, cohorts):
    df = pd.DataFrame(np.zeros(shape=(len(met_list_all), len(sample_list_all))),
                      index=met_list_all,
                      columns=sample_list_all)
    df.loc['cohort'] = pd.Series([''] * len(sample_list_all), index=sample_list_all, dtype='object')


    for c in cohorts:
        met_list    = cohorts[c]['met_list']
        sample_list = cohorts[c]['sample_list']
        matrix      = cohorts[c]["matrix"]

        df.loc[met_list, sample_list] = matrix.loc[met_list, sample_list]
        df.loc['cohort', sample_list] = c
        
    return df
    
def combine_cohort_datasets(sample_type='Normal'):
    cohorts = {
        "BRCA1": None,
        "CCRCC3": None,
        "CCRCC4": None,
        "COAD": None,
        "GBM": None,
        "HurthleCC": None,
        "PDAC": None,
        "PRAD": None,
    }

    for c in cohorts.keys():
        cohorts[c] = get_cohort_samples(c, sample_type)


    met_union_set = set()
    sample_list_all = []
    for c in cohorts.keys():
        met_union_set    |= set(cohorts[c]["met_list"])
        sample_list_all += cohorts[c]["sample_list"]

    met_list_all    = list(met_union_set)
    met_list_all.sort()

    df = construct_training_dataframe(met_list_all, sample_list_all, cohorts)

    return df, cohorts



def create_dataloaders(np_train_abun, np_train_lbls,
                       np_validation_abun=None, np_validation_lbls=None,
                       batch_size=32):
    train_dataset = MetaboliteDataset(
        np_mat=np_train_abun,
        cohort_labels=np_train_lbls
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )


    if np_validation_abun and np_validation_lbls:
        val_dataset  = MetaboliteDataset(
            np_mat=np_validation_abun,
            cohort_labels=np_validation_lbls
        )

        val_loader  = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        return train_loader, val_loader
    else:
        return train_loader