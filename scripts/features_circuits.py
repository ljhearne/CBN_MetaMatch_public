# %%
import numpy as np
import xarray as xr
from src.data_loader import result_loader
from statsmodels.stats.multitest import multipletests
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
from src.utils import array_to_mat


def get_mean_fc_values(mat, roi, roi_dict):
    all_values = []
    for hemi in ['lh', 'rh']:
        seed = roi_dict[roi][hemi]['subcortex']
        targets = roi_dict[roi][hemi]['cortex']
        values = mat[seed, :]
        values = values[targets]
        all_values.append(values)

    return np.nanmean(all_values)


def get_random_fc_values(mat, seed_dist=[0, 400], targ_dist=[0, 400],
                         n_targets=10):
    values_all = []
    for hemi in ['lh', 'rh']:
        seed = np.random.randint(seed_dist[0], high=seed_dist[1],
                                 size=1)
        targets = np.random.randint(targ_dist[0], high=targ_dist[1],
                                    size=n_targets)
        values = np.squeeze(mat[seed, :])
        values = values[targets]
        values_all.append(values)
    return np.nanmean(values_all)


def reorder_mat(mat, idx):
    # an interesting advanced indexing in python:
    # https://stackoverflow.com/questions/10936767/rearranging-matrix-elements-with-numpy
    new_mat = mat[:, idx][idx].copy()
    return new_mat


def feat_analysis_pathways(path_to_results, phenotype='group',
                           model='metamatch',
                           model_suffix='', denoise='scrubbing.5+gsr',
                           ksplit=5,
                           n_batch=200, n_batch_null=2000,
                           weight_type='prediction', n_targets=10):

    # Load haufe weights
    print('Loading weights...')
    weights_xr = result_loader(model=model, phenotype=phenotype,
                               model_suffix=model_suffix,
                               denoise=denoise,
                               ksplit=ksplit,
                               output_type='haufe_weights',
                               shuffled=False,
                               n_batch=n_batch, path=path_to_results)

    # Average across iterations and convert to numpy
    feature_weights = np.squeeze(weights_xr.sel(phenotype=phenotype,
                                                type=weight_type).to_array().to_numpy()).reshape(-1, ksplit*n_batch)

    # feature_weights_mean = weights_xr.sel(phenotype=phenotype,
    #                                       type=weight_type).mean(
    #                                           dim='iteration').to_array().to_numpy()
    feature_weights_mean = np.mean(feature_weights, axis=1)

    # Load null distribution haufe weights
    print('Loading shuffled weights...')
    null_weights_xr = result_loader(model=model, phenotype=phenotype,
                                    model_suffix=model_suffix,
                                    denoise=denoise,
                                    ksplit=ksplit,
                                    output_type='haufe_weights',
                                    shuffled=True,
                                    n_batch=n_batch_null, path=path_to_results)

    # convert to numpy
    null_weights = null_weights_xr.sel(
        phenotype=phenotype, type=weight_type).to_array().to_numpy().reshape(-1, ksplit*n_batch_null)
    null_weights = np.squeeze(null_weights)

    # roi list from Ye Tian's parcellation
    roi_list = ['NAc-shell', 'CAU-DA', 'PUT-DP', 'PUT-VA']

    # roi seeds from metamatching parcellation
    # these represent the 'nearest fit' given it doesn't
    # have the granularity to idenitify dorsal caudate and so on
    roi_seeds = [409, 418,
                 403, 413,
                 404, 414,
                 404, 414]

    # generate a dict that contains seed and target information
    roi_dict = {}
    c = 0
    for roi in roi_list:
        roi_dict[roi] = {}

        for hemi in ['lh', 'rh']:
            t_stat_txt = glob.glob(f"../results/HCP_roi/{roi}-{hemi}*sorted*")
            assert len(t_stat_txt) == 1
            cortical_roi = np.loadtxt(t_stat_txt[0],
                                      delimiter=',')[0:n_targets].astype(int)

            roi_dict[roi][hemi] = {}
            roi_dict[roi][hemi]['cortex'] = cortical_roi
            roi_dict[roi][hemi]['subcortex'] = roi_seeds[c]
            c += 1

    results = pd.DataFrame()
    for roi in roi_dict.keys():
        print(roi)

        # get the mean fc value from seed to targets,
        # averaging across hemisphere
        for i in tqdm(range(feature_weights.shape[1])):
            feat_mat = array_to_mat(feature_weights[:, i])
            feat_value = get_mean_fc_values(feat_mat, roi, roi_dict)
            _df = pd.DataFrame({
                'roi': [roi],
                'value': [feat_value],
                'model': ['model'],
                'iteration': i
            })
            results = pd.concat([results, _df], ignore_index=True)

        # permutation based null distribution
        for i in tqdm(range(null_weights.shape[1])):
            null_mat = array_to_mat(null_weights[:, i])
            feat_value = get_mean_fc_values(null_mat, roi, roi_dict)

            _df = pd.DataFrame({
                'roi': [roi],
                'value': [feat_value],
                'model': ['permutation']
            })
            results = pd.concat([results, _df], ignore_index=True)

    # convert features to matrix
    feature_weights_mat = array_to_mat(feature_weights_mean)
    for roi in roi_dict.keys():
        print(roi)

        # subcortex-cortex based null distribution
        for i in range(null_weights.shape[1]):
            feat_value = get_random_fc_values(feature_weights_mat,
                                              seed_dist=[400, 419],
                                              targ_dist=[0, 400],
                                              n_targets=n_targets)

            _df = pd.DataFrame({
                'roi': [roi],
                'value': [feat_value],
                'model': ['SC-Cort']
            })
            results = pd.concat([results, _df], ignore_index=True)

        # all cortex null distribution
        for i in range(null_weights.shape[1]):
            feat_value = get_random_fc_values(feature_weights_mat,
                                              seed_dist=[0, 400],
                                              targ_dist=[0, 400],
                                              n_targets=n_targets)

            _df = pd.DataFrame({
                'roi': [roi],
                'value': [feat_value],
                'model': ['Cort-Cort']
            })
            results = pd.concat([results, _df], ignore_index=True)

    # save the results out.
    out = (f"../results/feature_analysis/pathway_phenotype-{phenotype}_"
           f"model-{model}{model_suffix}_denoise-{denoise}_"
           f"n_batch-{n_batch}_n_batch_null-{n_batch_null}_"
           f"weight_type-{weight_type}_n_targets-{n_targets}")
    results.to_csv(out+'.csv', index=False)
    return results


if __name__ == "__main__":

    # parameters
    phenotype = 'group'
    model_suffix = '_site_confounds_removed'
    denoise = 'simple+gsr'
    n_batch = 200
    n_batch_null = 1000
    weight_type = 'prediction'
    path = '/home/lukeh/hpcworking/lukeH/projects/CBN_MetaMatch/results/kfold_july/'
    n_targets = 10

    print('Running pathway analysis...')
    for model in ['logreg', 'multilayer_metamatch']:
        _ = feat_analysis_pathways(path,
                                   phenotype=phenotype,
                                   model=model,
                                   model_suffix=model_suffix,
                                   denoise=denoise,
                                   ksplit=5,
                                   n_batch=n_batch,
                                   n_batch_null=n_batch_null,
                                   weight_type=weight_type,
                                   n_targets=n_targets)

