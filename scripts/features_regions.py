# %%
from abagen import get_expression_data
import pandas as pd
from src.data_loader import result_loader
import xarray as xr
from enigmatoolbox.permutation_testing import spin_test, shuf_test
import numpy as np
import numpy as np
from tqdm import tqdm
from src.utils import array_to_mat
from scipy.stats import spearmanr
import pickle


def spin_test_metamatch(a, b, n_rot=100, corr_type='pearson'):
    assert len(a) == len(b), "Input lengths do not match"
    assert len(a) == 400 or len(a) == 19, "Incorrect input length"

    # If surface data, do a spin test
    if len(a) == 400:
        p_spin, null_dist = spin_test(a, b, surface_name='fsa5',
                                      parcellation_name='schaefer_400',
                                      type=corr_type, n_rot=n_rot,
                                      null_dist=True)

    # If subcortical data, do a shuffle test
    elif len(a) == 19:
        p_spin, null_dist = shuf_test(a, b, n_rot=n_rot,
                                      type=corr_type,
                                      null_dist=True)

    return p_spin, null_dist


# parameters / glob
regions = [('surface', 0, 400), ('subcortex', 400, None)]
corr_type = 'spearman'
net_df = pd.read_csv(
    '../data/atlas/Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv')
network_labels = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn',
                  'Limbic', 'Cont', 'Default']

# parameters
phenotype = 'group'
model_suffix = '_site_confounds_removed'
model = 'multilayer_metamatch'
comparison_model = 'logreg'
denoise = 'simple+gsr'
n_batch = 200
n_batch_null = 1000
ksplit = 5
weight_type = 'prediction'
n_rot = 10000
path = '/home/lukeh/hpcworking/lukeH/projects/CBN_MetaMatch/results/kfold_july/'
outfile = f"../results/whole_brain_features/phenotype-{phenotype}_denoise-{denoise}_model-{model}{model_suffix}_Nrot-{str(n_rot)}_"


# Load haufe weights for both models
ds = []
model_list = [model, comparison_model]
for m in model_list:
    ds.append(result_loader(model=m, phenotype=phenotype,
                            model_suffix=model_suffix,
                            denoise=denoise, 
                            ksplit=ksplit,
                            output_type='haufe_weights',
                            n_batch=n_batch, path=path))
weights_xr = xr.concat(ds, dim='model')

# Load the null weights
ds = []
for m in model_list:
    ds.append(result_loader(model=m, phenotype=phenotype,
                            model_suffix=model_suffix, shuffled=True,
                            denoise=denoise,
                            ksplit=ksplit,
                            output_type='haufe_weights',
                            n_batch=n_batch_null, path=path))
weights_null_xr = xr.concat(ds, dim='model')

# Network analysis
print('Running network analysis...')
weights_all = np.squeeze(weights_xr.sel(phenotype=phenotype, type=weight_type
                                        ).to_array().to_numpy()).reshape(2, -1, ksplit*n_batch)
weights_all_null = np.squeeze(weights_null_xr.sel(phenotype=phenotype,
                                                  type=weight_type
                                                  ).to_array().to_numpy()).reshape(2, -1, ksplit*n_batch_null)
network_index = np.zeros((419))
for i, row in net_df.iterrows():
    for network, index in zip(['Vis', 'SomMot', 'DorsAttn',
                               'SalVentAttn', 'Limbic',
                               'Cont', 'Default', 'TempPar'],
                              [1, 2, 3, 4, 5, 6, 7, 7]):
        if network in row['ROI Name']:
            network_index[i] = index
network_index[-19::] = 8  # subcortex
network_labels.append('Subcortical')

# calculate weighted degree for each network / model
network_results = pd.DataFrame()

for i, m in enumerate(model_list):
    print(m)
    for n, label in tqdm(enumerate(network_labels)):
        index = network_index == (n+1)

        # Empirical values
        for it in range(weights_all.shape[2]):
            weighted_degree = np.nanmean(array_to_mat(weights_all[i, :, it]),
                                         axis=0)
            network_value = np.mean(weighted_degree[index])
            _results = pd.DataFrame()
            _results['network'] = [label]
            _results['model'] = m
            _results['permutation'] = False
            _results['iteration'] = it
            _results['value'] = network_value
            network_results = pd.concat([network_results, _results])

        # Null values
        for it in range(weights_all_null.shape[2]):
            weighted_degree = np.nanmean(array_to_mat(weights_all_null[i, :, it]),
                                         axis=0)
            network_value = np.mean(weighted_degree[index])
            _results = pd.DataFrame()
            _results['network'] = [label]
            _results['model'] = m
            _results['permutation'] = True
            _results['iteration'] = it
            _results['value'] = network_value
            network_results = pd.concat([network_results, _results])
network_results.to_csv(outfile+'network_results.csv')

# Region-level test against shuffled weights
# calculate weighted degree (nanmean is used to account for diag)
print('Region level analysis...')
# Average across iterations
mean_weights = weights_xr.sel(phenotype=phenotype, type=weight_type
                              ).mean(dim='iteration').mean('fold').to_array().to_numpy()
mean_weights = np.squeeze(mean_weights)

weighted_degree = {}
for i, m in enumerate(model_list):
    weighted_degree[m] = np.nanmean(array_to_mat(
        mean_weights[i, :]), axis=0)

# save out for later
with open(outfile+'weighted_degree.pickle', 'wb') as handle:
    pickle.dump(weighted_degree, handle, protocol=pickle.HIGHEST_PROTOCOL)

weighted_degree_null = {}
for i, m in enumerate(model_list):
    weighted_degree_null[m] = np.zeros((419, weights_all_null.shape[2]))
    for it in range(weights_all_null.shape[2]):
        weighted_degree_null[m][:, it] = np.nanmean(array_to_mat(
            weights_all_null[i, :, it]),
            axis=0)


# Contrast each parcel with a null permutation via percentile
results = pd.DataFrame()
for i, m in enumerate(model_list):
    for parcel in range(weighted_degree[m].shape[0]):

        value = weighted_degree[m][parcel]
        null_dist = weighted_degree_null[m][parcel, :]
        percentile = (null_dist < value).mean()

        # Express similar to a p value regardless of direction
        p_val = (0.5 - (abs(percentile - 0.5)))
    
        _results = pd.DataFrame()
        _results['parcel_index'] = [parcel]
        _results['model'] = m
        _results['percentile'] = percentile
        _results['p_val'] = percentile
        results = pd.concat([results, _results])
results.to_csv(outfile+'weighted_degree.csv')

# Comparison between models
print('Comparing weighted degree via spin test...')

results_list = []
for region_type, start_idx, end_idx in regions:

    # Extract data for the current regions
    a_data = weighted_degree[model][start_idx:end_idx]
    b_data = weighted_degree[comparison_model][start_idx:end_idx]

    # Perform the spin test
    p_spin, null_dist = spin_test_metamatch(a_data, b_data, n_rot=n_rot,
                                            corr_type=corr_type)
    # Perform a regular spearmanr as well
    r, p = spearmanr(a_data, b_data)

    # Print results
    print(region_type, ':', model, 'v.', comparison_model, 'p=', p_spin)

    # Append results as a dictionary to the list
    results_list.append({
        'model': model,
        'comparison_model': comparison_model,
        'r': r,
        'p': p,
        'p_spin': p_spin,
        'type': region_type
    })

# Convert the list of dictionaries to a DataFrame
results = pd.DataFrame(results_list)
results.to_csv(outfile+'weighted_degree_model_comparisons.csv')

## Gene expression
print('Analysing expression data...')

# get the expression data via abagen
# this is relatively slow
expression_data = get_expression_data(
    '/home/lukeh/hpcworking/shared/parcellations/Schaefer419_metamatching/FC419_MNI2mm.nii.gz', verbose=1, n_proc=4)

# list of implicated genes
genes = ['KIT', 'GRID2', 'WDR7', 'ADCK1']

# Save out the relevant gene data for plotting later
with open(outfile+'gene_expression.pickle', 'wb') as handle:
    pickle.dump(expression_data[genes], handle, protocol=pickle.HIGHEST_PROTOCOL)


results_list = []
for model in model_list:
    print(model)
    for gene in genes:
        for region_type, start_idx, end_idx in regions:

            # Extract data for the current region
            a_data = weighted_degree[model][start_idx:end_idx]
            b_data = expression_data[gene].values[start_idx:end_idx]

            # Perform the spin test
            p_spin, null_dist = spin_test_metamatch(a_data, b_data,
                                                    n_rot=n_rot,
                                                    corr_type=corr_type)
            # Perform a regular spearmanr as well
            idx = ~np.isnan(b_data)
            r, p = spearmanr(a_data[idx], b_data[idx])

            # Print results
            print(region_type, ':', gene, 'p=', p_spin)
            print('\t',r.round(2))
            # Append results as a dictionary to the list
            results_list.append({
                'a_label': model,
                'b_label': gene,
                'r': r,
                'p': p,
                'p_spin': p_spin,
                'type': region_type
            })

results = pd.DataFrame(results_list)
results.to_csv(outfile+'gene_comparisons.csv')


# %%
