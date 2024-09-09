import os
import xarray as xr
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from src.metamatch import run_kfold_logistic
from src.data_loader import load_model_inputs

# Create parser for options
parser = argparse.ArgumentParser(
    description='''Run the metamatch model''')

# These parameters must be passed to the function
parser.add_argument('--ds_list',
                    nargs='+',
                    default=None,
                    help='''list of datasets to use''')

parser.add_argument('--output',
                    type=str,
                    default=None,
                    help='''output file''')

parser.add_argument('--model',
                    type=str,
                    default=None,
                    help='''model to test''')

parser.add_argument('--denoise',
                    type=str,
                    default=None,
                    help='''denoise strat''')

parser.add_argument('--seed',
                    type=int,
                    default=0,
                    help='''''')

parser.add_argument('--phenotype_labels',
                    nargs='+',
                    default=None,
                    help='''''')

parser.add_argument('--confounds',
                    nargs='+',
                    default=None,
                    help='''''')

parser.add_argument('--k_iterations',
                    type=int,
                    default=100,
                    help='''''')

parser.add_argument('--k_splits',
                    type=int,
                    default=4,
                    help='''''')

parser.add_argument('--y_regress',
                    action='store_true',
                    help='''''')

parser.add_argument('--y_shuffle',
                    action='store_true',
                    help='''''')

parser.add_argument('--save_inputs',
                    action='store_true',
                    help='''''')

parser.add_argument('--save_weights',
                    action='store_true',
                    help='''''')

parser.add_argument('--perform_site',
                    action='store_true',
                    help='''''')

parser.add_argument('--resample',
                    type=int,
                    default=0,
                    help='''''')


# list of UKBB components (can be found on MM github)
UKBB_comp_list = ['Alcohol 1', 'Alcohol 2', 'Alcohol 3', 'Time walk', 'Time drive', 'Time TV ',
                  'Sleep', 'Age edu ', 'Work ', 'Travel', 'Household', 'Neuro',
                  'Hearing', 'Fluid Int.', 'Matching ', 'Sex ', 'Matching-o', 'Age',
                  'Trail-o C1', 'Trail-o C3', 'Digit-o C1', 'Digit-o C6', 'Sex G C1',
                  'Sex G C2', 'Genetic C1', 'Cancer C1', 'Urine C1', 'Blood C2',
                  'Blood C3', 'Blood C4', 'Blood C5', 'Deprive C1', 'Dur C1',
                  'Dur C2', 'Dur C4', 'Trail C1', 'Tower C1', 'Digit 1', 'Match',
                  'ProMem C1', '#Mem C1', 'Matrix C1', 'Matrix C2', 'Matrix C3',
                  'Illness C1', 'Illness C4', 'Loc C1', 'Breath C1', 'Grip C1',
                  'ECG C1', 'ECG C2', 'ECG C3', 'ECG C6', 'Carotid C1', 'Carotid C5',
                  'Bone C1', 'Bone C3', 'Body C1', 'Body C2', 'Body C3', 'BP eye C2',
                  'BP eye C3', 'BP eye C4', 'BP eye C5', 'BP eye C6', 'Family C1',
                  'Smoke C1']


def run_model(ds_list, output, model, denoise, seed, phenotype_labels, confounds,
              perform_site=False, y_regress=False, k_iterations=100, k_splits=10,
              y_shuffle=False, save_inputs=False, save_weights=False,
              resample=0):

    # Get model inputs
    y_df, X = load_model_inputs(ds_list, denoise)

    # ml for handedness
    y_df['handedness'].replace({'right': 0, 'left': 1}, inplace=True)
    
    # perform resampling
    if resample == 0:
        # do nothing
        pass
    else:
        print('resampling data, N =', resample)
        y_df['manual_index'] = range(len(y_df))

        _, index = train_test_split(range(len(y_df)), test_size=resample,
                                    stratify=y_df[["group"]])
        
        # sort the index
        index = sorted(index)

        # rewrite inputs with new index
        y_df = y_df[y_df.manual_index.isin(index)]
        X = X[index, :]

        # also, don't save weights
        save_weights = False

    # add site
    site_data = y_df['site'].values

    # select phenotype data
    y = y_df[phenotype_labels].values

    # select confound data
    y_confounds = y_df[confounds].values

    # assert only one phenotype
    assert y.shape[1] == 1, "more than 1 phenotype"

    # print the data shapes
    print('Input data shapes:')
    print('X:', X.shape)
    print('y:', y.shape, phenotype_labels)
    print('confounds:', y_confounds.shape, confounds)

    # run the kfold
    predictions, haufe_weights, reduced_weights = run_kfold_logistic(
        X, y, phenotype_labels, site_data, k_iterations=k_iterations, model=model,
        k_splits=k_splits, y_regress=y_regress, y_shuffle=y_shuffle,
        y_control=y_confounds, perform_site=perform_site, seed=seed)

    print('Saving outputs...')
    if resample == 0:
        output = output+'_'+phenotype_labels[0]
    else:
        print('adding resample tag to output')
        output = output+'_'+phenotype_labels[0]+'_N'+str(resample)

    # save the inputs, predictions and weights:
    predictions.to_netcdf(output+'_predictions.nc')

    # save inputs if flagged.
    if save_inputs:
        print('Saving inputs...')
        model_inputs = xr.Dataset({'y': (['participant_id', 'phenotype'], y),
                                   'X': (['participant_id', 'edge'], X),
                                   'y_confounds': (['participant_id',
                                                    'confound'], y_confounds),
                                   'site': (['participant_id'],
                                            y_df['site'].values.reshape(-1))},
                                  coords={'phenotype': phenotype_labels,
                                          'confound': confounds})

        model_inputs.to_netcdf(output+'_inputs.nc')

    # save weights after averaging across folds to save space
    if save_weights:
        print('Saving weights...')
        haufe_weights.to_netcdf(output+'_haufe_weights.nc')

        if model == "multilayer_metamatch":
            reduced_weights.to_netcdf(output+'_reduced_weights.nc')

    return None


if __name__ == '__main__':
    # Read in user-specified parameters
    args = parser.parse_args()

    # run the ML model
    run_model(args.ds_list, args.output, args.model, args.denoise, args.seed,
              args.phenotype_labels, args.confounds, args.perform_site,
              args.y_regress, args.k_iterations, args.k_splits, 
              args.y_shuffle, args.save_inputs, args.save_weights, 
              args.resample)
