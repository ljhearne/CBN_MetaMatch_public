from src.generate_subject_lists import generate_subject_lists
from src.utils import search_bids, nedge_to_nnode
import argparse
import xarray as xr
import pandas as pd
import numpy as np
import json

mm_n_edges = 87571  # the number of edges in the mm model
mm_n_components = 67  # the number of predictions made in the mm model

# Create parser for options
parser = argparse.ArgumentParser(
    description='''Run the metamatch model''')

# These parameters must be passed to the function
parser.add_argument('--config_file',
                    type=str,
                    default=None,
                    help='''input config file''')

parser.add_argument('--denoise',
                    type=str,
                    default=None,
                    help='''denoise strat''')

parser.add_argument('--space',
                    type=str,
                    default='MNI152NLin2009cAsym',
                    help='''img space''')

parser.add_argument('--fd_thresh',
                    type=float,
                    default=0.5,
                    help='''fd magnitude threshold for subj selection''')

parser.add_argument('--fd_time',
                    type=int,
                    default=5,
                    help='''fd time (minutes) threshold for subj selection''')


def prepare_model_inputs(study_label, bids_dir, fmriprep_dir, conn_dir, 
                         phenotype_list, denoise, space, session, group, 
                         group_label, site, fd_thresh, fd_time):
    '''
    This function organises the model inputs (phenotype and fc data)
    '''

    # Generate subject lists based on head motion
    subject_list, subject_list_id, mean_fd = generate_subject_lists(
        fmriprep_dir, space=space, session=session, fd_thresh=fd_thresh,
        fd_time=fd_time, group=group)

    # Phenotypic inputs:
    layout = search_bids(conn_dir, bids_dir)

    # get all phenotypes (y)
    phenotype_df = layout.glob_phenotype_one_to_one(session=session)

    # subject filter (row)
    idx = phenotype_df.index.isin(['sub-' + s for s in subject_list])
    phenotype_df = phenotype_df.iloc[idx, :]

    # add framewise displacment
    phenotype_df['avg_framewise_displacement'] = mean_fd.copy()
    
    # filter columns by phenotype
    phenotype_df = phenotype_df.loc[:, phenotype_list]

    # Prepare df for machine learning by using integers in place of strings
    phenotype_df['study'] = study_label
    phenotype_df['site'] = site
    phenotype_df['site'].replace({'Brisbane': 1, 'Melbourne': 2, 'Korea': 3},
                                 inplace=True)
    phenotype_df['group'] = group_label
    phenotype_df['group'].replace({'patient': 1, 'control': 0}, inplace=True)
    phenotype_df['gender'].replace({'male': 1, 'female': 0}, inplace=True)

    # harmonise column names across datasets
    for old, new in zip(['fsiq-4_comp_score',
                        'WASI_II_FSIQ4_Composite_Score',
                         'P1_BSL_YBOCS_total',
                         'Age at Beginning of Phase 1',
                         'P1_BSL_HAMA_ Total', 'HAMA',
                         'HAMD', 'P1_BSL_MADRS_ Total', 'education'],
                        ['IQ', 'IQ', 'ybocs_total', 'age', 'hama_total',
                         'hama_total', 'hamd_total', 'madrs_total',
                         'years_of_education']):
        if old in phenotype_df.columns:
            # Rename the 'old_column_name' to 'new_column_name'
            phenotype_df.rename(columns={old: new}, inplace=True)

    # Functional connectivity inputs:
    input_files = []
    for subj in layout.get_subjects():
        # get file connectivity csv
        f = layout.file_filter({'task': 'rest',
                                'ses': session,
                                'space': space,
                                'atlas': 'Schaefer419',
                                'denoise': denoise+'_',
                                'fcmethod': 'correlation',
                                'extension': 'csv'},
                               subjects=[subj])
        assert len(f) == 1, "Error in returning a single FC matrix per subject:"
        input_files.append(f[0])

    # assert that the subjects in the input files and the phenotype
    # df match
    phenotype_df['input_file'] = np.nan
    for i, sid in enumerate(phenotype_df.index.unique()):

        # ensure that the phenotype data and FC files match up
        input = []
        for f in input_files:
            sid_compare = 'sub-'+f.split('/sub-')[-1].split('_')[0]
            if sid == sid_compare:
                input.append(f)

        assert len(input) == 1, "Error in matching phenotype and FC IDs"
        phenotype_df.loc[phenotype_df.index == sid, 'input_file'] = input[0]

    # # drop participants with any missing phenotype data
    # print('N dropped rows:',
    #       phenotype_df.shape[0] - phenotype_df.dropna(axis=0).shape[0])
    # phenotype_df = phenotype_df.dropna(axis=0)

    # Functional connectivity inputs
    fc_all = np.zeros((phenotype_df.shape[0], mm_n_edges))
    for i, (_, row) in enumerate(phenotype_df.iterrows()):

        # take lower triangle
        # identical index to metamatch github repo
        fc = np.loadtxt(row['input_file'], delimiter=',')
        fc_all[i, :] = fc[np.tril(
            np.ones(nedge_to_nnode(mm_n_edges)), k=-1) == 1]

    # check the shapes of the data
    assert fc_all.shape[1] == mm_n_edges, "X is wrong shape"
    assert fc_all.shape[0] == phenotype_df.shape[0], "Samples do not match in X and y"

    # Save to xarray and csv.
    phenotype_df.to_csv(
        f"../data/model_inputs/{study_label}_denoise-{denoise}.csv")
    xr_df = pd.DataFrame.to_xarray(phenotype_df)
    xr_X = xr.DataArray(fc_all,
                        dims=('participant_id', 'fc'),
                        coords={'participant_id': xr_df['participant_id']})
    xr_X.to_netcdf('../data/model_inputs/'+study_label +
                   '_denoise-'+denoise+'_fc.nc')
    return None


def iterate_config(config_file, denoise, space, fd_thresh, fd_time):
    # read in the JSON config file...
    with open(config_file) as f:
        dataset_dict = json.load(f)

    # for each dataset, prepare model inputs
    for key, params in dataset_dict.items():
        print('Running', key, 'ds dict')
        prepare_model_inputs(params['study_label'],
                             params['bids_dir'], params['fmriprep_dir'],
                             params['conn_dir'], params['phenotype_list'],
                             denoise, space, params['session'],
                             params['group'], params['group_label'],
                             params['site'], fd_thresh, fd_time)
    return None


if __name__ == '__main__':
    # Read in user-specified parameters
    args = parser.parse_args()

    # run
    iterate_config(args.config_file, args.denoise,
                   args.space, args.fd_thresh, args.fd_time)
