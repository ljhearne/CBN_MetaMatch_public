# custom scripts to load data
import xarray as xr
import pandas as pd
import numpy as np


def result_loader(model='metamatch', model_suffix='',
                  phenotype='group', denoise='simple+gsr',
                  ksplit=10, output_type='predictions',
                  n_batch=1, shuffled=False, resample=0,
                  path='/home/lukeh/hpcworking/lukeH/projects/CBN_MetaMatch/results/kfold/'):
    # little helper function to load results
    files = []
    for batch in range(n_batch):
        # organise file name
        if shuffled:
            model_path = f"{path}/{model}_SHUFFLED/"
        else:
            model_path = f"{path}/{model}/"

        file_name = (f"kfold_denoise-{denoise}_ksplit-{ksplit}_"
                     f"model-{model}{model_suffix}_batch-{batch}_{phenotype}_")

        if resample != 0:
            file_name = f"{file_name}N{resample}_"

        # add to list
        files.append(f"{model_path}{file_name}{output_type}.nc")

    # combine across iterations
    weights = xr.open_mfdataset(
        files, concat_dim='iteration', combine='nested')
    weights = weights.assign_coords(model=model+model_suffix)
    return weights


def load_model_inputs(ds_list, denoise, path_to_data='../data/model_inputs/'):
    # Init fc array
    # had issues with memory using get_model_inputs
    dataset = xr.open_dataset(path_to_data+ds_list[0]
                              + '_denoise-'+denoise+'_fc.nc')
    fc = np.zeros((0, dataset.dims['fc']))
    dataset.close()

    phen_dfs = []
    for ds in ds_list:

        # load fc
        f = path_to_data+ds+'_denoise-'+denoise+'_fc.nc'
        dataset = xr.open_dataset(f)
        fc_data = np.squeeze(dataset.to_array().to_numpy())

        data_ids = dataset.participant_id.to_dict()['data']
        dataset.close()

        # load phenotype
        f = path_to_data+ds+'_denoise-'+denoise+'.csv'
        df = pd.read_csv(f)

        # assert the ids match
        assert df['participant_id'].to_list(
        ) == data_ids, "ids don't match during loading"

        # combine data after assert
        phen_dfs.append(df)
        fc = np.vstack((fc, fc_data))

    df = pd.concat(phen_dfs, ignore_index=True)
    assert df.shape[0] == fc.shape[0]
    return df, fc
