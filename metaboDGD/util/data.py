import pandas as pd
import numpy  as np

def get_cohort_samples(cohort_name):
    # Get the xls file
    xls = pd.ExcelFile(f'data/PreprocessedData_{cohort_name}.xlsx')

    # Get the dataframes for the preprocessed metabolomics data
    n = xls.parse("metabo_imputed_filtered_Normal")

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
    
    for c in cohorts:
        met_list    = cohorts[c]['met_list']
        sample_list = cohorts[c]['sample_list']
        matrix      = cohorts[c]["matrix"]

        df.loc[met_list, sample_list] = matrix.loc[met_list, sample_list]
        
    return df
    
def combine_cohort_datasets():
    cohorts = {
        "BRCA1": None,
        "ccRCC3": None,
        "ccRCC4": None,
        "COAD": None,
        "GBM": None,
        "HurthleCC": None,
        "PDAC": None,
        "PRAD": None,
    }

    for c in cohorts.keys():
        cohorts[c] = get_cohort_samples(c)


    met_union_set = set()
    sample_list_all = []
    for c in cohorts.keys():
        met_union_set    |= set(cohorts[c]["met_list"])
        sample_list_all += cohorts[c]["sample_list"]

    met_list_all    = list(met_union_set)
    met_list_all.sort()

    df = construct_training_dataframe(met_list_all, sample_list_all, cohorts)

    return df, cohorts

def create_dataloaders():
    pass