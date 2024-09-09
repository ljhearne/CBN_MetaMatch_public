# %%
#import numpy as np
import xarray as xr
from src.data_loader import result_loader
from statsmodels.stats.multitest import multipletests
import pandas as pd
import numpy as np
#import glob
#from tqdm import tqdm
#from src.utils import array_to_mat


def feat_analysis_UKBB(path_to_results, phenotype='group', model='metamatch',
                       model_suffix='', denoise='scrubbing.5+gsr', ksplit=5,
                       n_batch=200, n_batch_null=2000):

    # Load the UKBB weights into a dataframe
    weights_df = result_loader(model=model, phenotype=phenotype,
                               model_suffix=model_suffix,
                               denoise=denoise,
                               ksplit=ksplit,
                               output_type='reduced_weights',
                               n_batch=n_batch, path=path_to_results
                               ).to_dataframe().reset_index()

    null_weights_df = result_loader(model=model, phenotype=phenotype,
                                    model_suffix=model_suffix,
                                    denoise=denoise,
                                    ksplit=ksplit,
                                    shuffled=True,
                                    output_type='reduced_weights',
                                    n_batch=n_batch_null, path=path_to_results
                                    ).to_dataframe().reset_index()

    # add component labels if using multilayer metamatching
    if weights_df.component.max() == 457:
        print("multilayer results detected")
        with open("../docs/multilayer_components.txt") as file:
            component_list = [line.rstrip() for line in file]

        weights_df['component_index'] = weights_df['component'].copy()
        null_weights_df['component_index'] = null_weights_df['component'].copy()

        for i in range(weights_df.component_index.max()+1):
            weights_df.loc[weights_df.component_index ==
                           i, "component"] = component_list[i]
            null_weights_df.loc[null_weights_df.component_index ==
                                i, "component"] = component_list[i]

    # Combine the two dataframes
    weights_df['permutation'] = False
    null_weights_df['permutation'] = True
    weights_df = pd.concat([weights_df, null_weights_df])
    weights_df = weights_df.rename(columns={
        '__xarray_dataarray_variable__': 'value'}).reset_index()

    # Statistical inference
    percentiles = []
    p_values = []

    for component in weights_df.component.unique():
        mean_value = weights_df.loc[(weights_df.component == component)
                                    & (weights_df.permutation == False)
                                    ].value.mean()

        null_distribution = weights_df.loc[(weights_df.component == component)
                                           & (weights_df.permutation == True)
                                           ].value.values

        # Calculate percentile
        perc = (null_distribution < mean_value).mean()
        percentiles.append(perc)

        # Express similar to a p value regardless of direction
        p_values.append(0.5 - (abs(perc - 0.5)))

    # Perform multiple comparison correction
    p_values_fdr = multipletests(p_values, method='fdr_bh', alpha=0.05)

    # Put the values into the dataframe
    weights_df['percentile'] = pd.NA
    weights_df['p'] = pd.NA
    weights_df['p_fdr'] = pd.NA
    for i, component in enumerate(weights_df.component.unique()):
        idx = weights_df.loc[(weights_df.component == component)
                             & (weights_df.permutation == False)].index
        weights_df.loc[idx, 'percentile'] = percentiles[i]
        weights_df.loc[idx, 'p'] = p_values[i]
        weights_df.loc[idx, 'p_fdr'] = p_values_fdr[1][i]

    # Save
    # save the results out.
    out = (f"../results/feature_analysis/UKBB_phenotype-{phenotype}_"
           f"model-{model}{model_suffix}_denoise-{denoise}_"
           f"n_batch-{n_batch}_n_batch_null-{n_batch_null}")
    weights_df.to_csv(out+'.csv', index=False)
    return weights_df


if __name__ == "__main__":

    # parameters
    phenotype = 'group'
    model_suffix = '_site_confounds_removed'
    model = 'multilayer_metamatch'
    denoise = 'simple+gsr'
    n_batch = 200
    n_batch_null = 2000
    path = '/home/lukeh/hpcworking/lukeH/projects/CBN_MetaMatch/results/kfold_july/'

    _ = feat_analysis_UKBB(path,
                           phenotype=phenotype,
                           model=model,
                           model_suffix=model_suffix, 
                           denoise=denoise,
                           ksplit=5,
                           n_batch=n_batch, n_batch_null=n_batch_null)

# %%
