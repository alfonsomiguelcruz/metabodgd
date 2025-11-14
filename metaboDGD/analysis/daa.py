import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, mannwhitneyu

def perform_differential_abundance_analysis():
    pass


def get_daa_mwu_metrics(grp_one, grp_two, metabolite_list, fdr_method='bh'):
    p_value = mannwhitneyu(grp_one, grp_two).pvalue
    q_value = false_discovery_control(p_value, method=fdr_method)
    log2_fc = np.abs(np.mean(grp_one + 1e-6, axis=0) - np.mean(grp_two + 1e-6, axis=0))

    return pd.DataFrame({
        'p_value': p_value,
        'q_value': q_value,
        'log2_fold_change': log2_fc
    }, index=metabolite_list)