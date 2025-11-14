import pickle
import pandas as pd
import numpy  as np
from torch.utils.data import DataLoader
from metaboDGD.src.dataset import MetaboliteDataset



def get_cohort_samples(cohort_name, sample_type='Normal'):
    """
    Gets the metabolite abundance data from all samples
    of each cohort.

    Parameters
    ----------
    cohort_name : `str`
        A string for the name of the cohort to get samples
        from.

    sample_type : `str`, default="Normal"
        A string for the type of samples to get. These may
        either be normal or tumor samples.


    Returns
    -------
    cohort dictionary : `dict`
        A dictionary containing the sample labels, list of
        identified metabolites, and the matrix of
        metabolite abundances of samples in the cohort.  
    """
    
    # Get the xls file
    xls = pd.ExcelFile(f'data/PreprocessedData_{cohort_name}.xlsx')

    # Get the dataframes for the preprocessed metabolomics data
    n = xls.parse(f"metabo_imputed_filtered_{sample_type}")

    # Get list of metabolites
    n_met_list = n["Unnamed: 0"].to_list()

    # Get list of sample IDs
    n_sample_list = n.columns.to_list()[1:]

    # Update the sample list names with cohort name to prevent duplication
    n_sample_list = [cohort_name + '_' + s for s in n_sample_list]

    # Set Index to metabolite names, drop "Unnamed: 0" column
    n.set_index("Unnamed: 0", inplace=True)

    # Update the column names with cohort names, consistent with n_sample_list
    n.columns = [cohort_name + '_' + c for c in n.columns]

    return {
        "sample_list"   : n_sample_list,
        "met_list"      : n_met_list,
        "matrix"        : n
    }


def construct_training_dataframe(met_list_all, sample_list_all, cohorts):
    """
    Creates a dataframe that combines all metabolites and
    samples from all cohorts.

    Parameters
    ----------
    met_list_all : `list`
        A list of identified metabolites across all
        cohorts.

    sample_list_all : `list`
        A list of identifiers of all samples across
        all cohorts.

    cohorts : `dict`
        A dictionary containing the cohort dictionaries of
        each cohort.


    Returns
    -------
    df : `pandas.DataFrame`
        A dataframe of metabolite abundances of all
        identified metabolites for all samples. 
    """

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
    """
    Combines and preprocesses the datasets of all cohorts.

    Parameters
    ----------
    sample_type : `str`, default="Normal"
        A string for the type of samples to get. These may
        either be normal or tumor samples.


    Returns
    -------
    df : `pandas.DataFrame`
        A dataframe of metabolite abundances of all
        identified metabolites for all samples. 

    cohorts : `dict`
        A dictionary of dictionaries. Each entry is a
        dictionary containing the sample labels, list of
        identified metabolites, and the matrix of
        metabolite abundances of samples of a cohort.  
    """

    cohorts = {
        "BRCA1": None,
        "CCRCC3": None,
        "CCRCC4": None,
        "COAD": None,
        "GBM": None,
        "HurthleCC": None,
        "PDAC": None,
        "PRAD": None,
        "feces_MTBLS6334": None,
        "feces_MTBLS7866": None,
        "plasma_MTBLS11094": None,
        "plasma_MTBLS11656": None,
        "plasma_MTBLS11746": None,
        # "plasma_MTBLS1183": None,
        "plasma_MTBLS11996": None,
        "plasma_MTBLS2262": None,
        "plasma_MTBLS3305": None,
        "plasma_MTBLS8390": None,
        "saliva_MTBLS4569": None,
        "saliva_MTBLS760": None,
        "saliva_MTBLS7807": None,
        "serum_MTBLS12328": None,
        "serum_MTBLS12539": None,
        "serum_MTBLS12576": None,
        "serum_MTBLS1839": None,
        "serum_MTBLS2615": None,
        "serum_MTBLS3838": None,
        "serum_MTBLS6039": None,
        "serum_MTBLS6982": None,
        "serum_MTBLS7878": None,
        "serum_MTBLS8644": None,
        "tissue_MTBLS1122": None,
    }

    if type(sample_type) == list:
        st_disease, st_tumor = sample_type
    
    for c in cohorts.keys():
        if sample_type == 'Normal':
            cohorts[c] = get_cohort_samples(c, sample_type)
        elif any(bio in c for bio in ['feces', 'plasma', 'saliva', 'serum', 'tissue']):
            cohorts[c] = get_cohort_samples(c, st_disease)
        else:
             cohorts[c] = get_cohort_samples(c, st_tumor)
        


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
    """
    Creates training and validation datasets from the
    metabolite abundance data.

    Parameters
    ----------
    np_train_abun : `numpy.ndarray`
        Numpy array containing the metabolite abundance
        data for the training set.

    np_train_lbls : `list`
        A list of strings containing the cohorts the samples
        originally belonged to.

    np_validation_abun : `numpy.ndarray`, default=None
        Numpy array containing the metabolite abundance
        data for the validation set.

    np_validation_lbls : `list`, default=None
        A list of strings containing the cohorts the samples
        originally belonged to.

    batch_size : `int`
        Number of samples stored per batch in training the
        model.


    Returns
    -------
    train_loader : `torch.utils.data.DataLoader`
        A `DataLoader` object that stores the metabolite
        abundance data of the training set with a
        `MetaboliteDataset` object.

    val_loader : `torch.utils.data.DataLoader`
        A `DataLoader` object that stores the metabolite
        abundance data of the validation set with a
        `MetaboliteDataset` object.
    """

    train_dataset = MetaboliteDataset(
        np_mat=np_train_abun,
        cohort_labels=np_train_lbls
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )


    if np_validation_abun is not None and np_validation_lbls is not None:
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
    

def load_dataframe_and_cohort(dir,
                              dataframe_fname,
                              cohort_fname):
    # Read the sample dataframe and convert to a numpy array
    df = pd.read_csv(dir + dataframe_fname)
    df.set_index('Unnamed: 0', inplace=True)
    df.index.name = None
    np_lbls = df.loc['cohort'].to_numpy()
    np_raw_lbls = np.copy(np_lbls)
    np_log = df.T.drop(columns=['cohort']).astype('float32').to_numpy()

    np_lbls = np.where(np.isin(np_lbls,
    [
        'feces_MTBLS6334',
        'feces_MTBLS7866'
    ]),
    'feces', np_lbls)

    np_lbls = np.where(np.isin(np_lbls,
        [
            'plasma_MTBLS11094',
            'plasma_MTBLS11746',
            'plasma_MTBLS11656',
            'plasma_MTBLS3305',
            'plasma_MTBLS8390',
            'plasma_MTBLS2262',
            'plasma_MTBLS11996'
        ]),
        'plasma', np_lbls)

    np_lbls = np.where(np.isin(np_lbls,
        [
            'saliva_MTBLS4569',
            'saliva_MTBLS7807',
            'saliva_MTBLS760'
        ]),
        'saliva', np_lbls)

    np_lbls = np.where(np.isin(np_lbls,
        [
            'serum_MTBLS12539',
            'serum_MTBLS12576',
            'serum_MTBLS8644',
            'serum_MTBLS7878',
            'serum_MTBLS6982',
            'serum_MTBLS2615',
            'serum_MTBLS1839',
            'serum_MTBLS12328',
            'serum_MTBLS6039',
            'serum_MTBLS3838'
        ]),
        'serum', np_lbls)
    
    np_lbls = np.where(np.isin(np_lbls,
        [
            'tissue_MTBLS1122',
        ]),
        'tissue', np_lbls)

    # Read the sample cohort dictionary
    fn = open(dir + cohort_fname, 'rb')
    cohorts = pickle.load(fn)
    fn.close()

    return np_lbls, np_raw_lbls, np_log, cohorts