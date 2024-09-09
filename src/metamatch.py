'''
This code performs the metamatching procedure.
It closely mimics the official demo code released
on the MM repo.
'''
import os
import torch
import numpy as np
import pandas as pd
import xarray as xr
import pickle
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from src.CBIG_model_pytorch import demean_norm, multi_task_dataset, covariance_rowwise
from neuroHarmonize import harmonizationLearn, harmonizationApply


# parameters that should not need to change
batch_size = 16
device = 'cpu'
mm_n_edges = 87571  # the number of edges in the mm model (this is fixed)
# the number of predictions made in the mm model (UKBB components, also fixed)
mm_n_components = 67
ml_mm_n_components = 458

# path to the metamatching model and where the data are stored
# (after it has been prepared in prepare_metamatch_model.py)
path_repo = '../data/Meta_matching_models-main/'
path_model_weight = os.path.join(
    path_repo, 'v1.1', 'meta_matching_v1.1_model_scripted.pt')
path_model_weight_multilayer = os.path.join(
    path_repo, 'v2.0/models/', 'meta_matching_v2.0_model_scripted.pt')


def get_multilayer_mm_predictions(
    x,
    y,
    dataset_names={
        'extra-large': 'UKBB',
        'large': 'ABCD',
        'medium': ['GSP', 'HBN', 'eNKI']
    },
    n_cpus=1
):
    '''Predict using multilayer meta-matching models
    
    Args:
        x (ndarray): input FC data
        y (ndarray): target phenotype label
        dataset_names (dict): names of extra-large, large, medium source datasets 
    
    Returns:
        ndarray: prediction on x from multilayer metamatching models
    '''
    model_path = f"{path_repo}v2.0/models/"
    torch.set_num_threads(n_cpus)

    # load the pretrained network
    net = torch.jit.load(path_model_weight_multilayer, map_location=device)
    net.to(device)
    net.train(False)

    # Prepare data for DNN
    n_subj = x.shape[0]
    batch_size = 16
    y_dummy = np.zeros(y.shape)
    dset = multi_task_dataset(x, y_dummy, True)
    dataLoader = DataLoader(dset, batch_size=batch_size,
                            shuffle=False, num_workers=1)

    # Phenotypic prediction from extra-large source dataset
    dataset_XL = dataset_names['extra-large']
    n_phe_dict = {}
    models = pickle.load(open(os.path.join(model_path, dataset_XL
                                           + '_rr_models.sav'), 'rb'))
    n_phe_dict[dataset_XL] = len(models.keys())

    y_pred_dnn_XL = np.zeros((0, n_phe_dict[dataset_XL]))
    for (x_batch, _) in dataLoader:
        x_batch = x_batch.to(device)
        outputs = net(x_batch)
        y_pred_dnn_XL = np.concatenate((y_pred_dnn_XL,
                                        outputs.data.cpu().numpy()), axis=0)

    y_pred_rr_XL = np.zeros((x.shape[0], n_phe_dict[dataset_XL]))
    for phe_idx, phe_name in enumerate(models):
        y_pred_rr_XL[:, phe_idx] = models[phe_name].predict(x)

    y_pred_rr_1layer = {}
    y_pred_rr_2layer = {}

    # Phenotypic prediction from large source dataset
    dataset_L = dataset_names['large']
    models_1layer = pickle.load(open(os.path.join(model_path, dataset_L
                                                  + '_rr_models_base.sav'),
                                     'rb'))
    models_2layer = pickle.load(open(os.path.join(model_path, dataset_L
                                                  + '_rr_models_multilayer.sav'),
                                     'rb'))
    n_phe_dict[dataset_L] = len(models_1layer.keys())
    y_pred_rr_1layer[dataset_L] = np.zeros((n_subj, n_phe_dict[dataset_L]))
    y_pred_rr_2layer[dataset_L] = np.zeros((n_subj, n_phe_dict[dataset_L]))

    for phe_idx, phe_name in enumerate(models_1layer):
        y_pred_rr_1layer[dataset_L][:, phe_idx] = models_1layer[phe_name
                                                                ].predict(x)

    for phe_idx, phe_name in enumerate(models_2layer):
        x_stacking = np.concatenate((y_pred_dnn_XL, y_pred_rr_XL), axis=1)
        y_pred_rr_2layer[dataset_L][:, phe_idx] = models_2layer[phe_name
                                                                ].predict(x_stacking)

    # Phenotypic prediction from medium source dataset
    for dataset_M in dataset_names['medium']:
        models_1layer = pickle.load(open(os.path.join(model_path, dataset_M
                                                      + '_rr_models_base.sav'), 'rb'))
        models_2layer = pickle.load(open(os.path.join(model_path, dataset_M
                                                      + '_rr_models_multilayer.sav'), 'rb'))
        n_phe = n_phe_dict[dataset_M] = len(models_1layer.keys())
        y_pred_rr_1layer[dataset_M] = np.zeros((n_subj, n_phe))
        y_pred_rr_2layer[dataset_M] = np.zeros((n_subj, n_phe))

        for phe_idx, phe_name in enumerate(models_1layer):
            y_pred_rr_1layer[dataset_M][:, phe_idx] = models_1layer[phe_name
                                                                    ].predict(x)

        for phe_idx, phe_name in enumerate(models_2layer):
            x_stacking = np.concatenate((y_pred_dnn_XL, y_pred_rr_XL,
                                         y_pred_rr_1layer[dataset_L]), axis=1)

            y_pred_rr_2layer[dataset_M][:, phe_idx] = models_2layer[phe_name
                                                                    ].predict(x_stacking)

    y_pred = np.concatenate([y_pred_dnn_XL] + [y_pred_rr_XL] +
                            list(y_pred_rr_1layer.values()) +
                            list(y_pred_rr_2layer.values()), axis=1)
    return y_pred


def get_mm_predictions(X, y, n_cpus=1):
    '''
    Runs input X through the metamatching neural
    network based on UKBB. It is essentially
    downsampling the large number of edges to
    67.

    Parameters
    ----------
    X
        an array of participants by edges (87571)
    y
        an array of participants by phenotypes
    n_cpus, optional
        number of cpus, by default 1

    Returns
    -------
        returns the 67 x phenotypes predictions
    '''
    torch.set_num_threads(n_cpus)

    # load the pretrained network
    net = torch.jit.load(path_model_weight, map_location=device)
    net.to(device)
    net.train(False)

    # test data is set to 0 as these are unknown (the 67 biobank variables)
    dset_train = multi_task_dataset(X, np.zeros(y.shape), True)
    dloader = DataLoader(dset_train,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=1)

    # create training predictions from pretrained network
    # 67 can be hardcoded as the metamatch model never changes
    y_pred = np.zeros((0, mm_n_components))
    for (x, _) in dloader:
        x = x.to(device)
        outputs = net(x)
        y_pred = np.concatenate((y_pred, outputs.data.cpu().numpy()), axis=0)
    return y_pred


def confound_regression(X_train, X_test, y_train, y_test):
    # use linear regression to remove confounding variables
    # (X) from the dependent variable (y) in the training
    # set. Then, apply this model to the test set.
    
    # Fit a linear regression model to training X and Y
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Get the predicted values based on X
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Subtract the predicted values from Y to get the residuals
    # (effect of confounding variable removed), also add the
    # intercept
    train_residuals = y_train - y_train_pred + model.intercept_
    test_residuals = y_test - y_test_pred + model.intercept_

    return train_residuals, test_residuals


def site_harmonisation(site_train, site_test,
                       X_train, X_test,
                       y_train, y_test):
    # performs site harmonisation via neuroCombat
    # organise input dataframes
    train_covars = pd.DataFrame({'SITE': site_train,
                                 'y': y_train.reshape(-1)})
    test_covars = pd.DataFrame({'SITE': site_test,
                                'y': y_test.reshape(-1)})

    model, train_data_adj = harmonizationLearn(data=X_train, 
                                               covars=train_covars)
    test_data_adj = harmonizationApply(X_test,
                                       test_covars,
                                       model)

    return train_data_adj, test_data_adj


def log_model(X_train, y_train, X_test):

    # for cross val
    Cs = [1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1, 10]

    # preallocate outputs
    y_pred = np.zeros((X_test.shape[0], y_train.shape[1]))
    y_pred_prob = np.zeros((y_pred.shape))
    y_train_pred = np.zeros((y_train.shape))
    y_train_pred_prob = np.zeros((y_train.shape))

    # loop through the phenotypes
    # (a seperate model is created for each one)
    for i in range(y_train.shape[1]):

        # create model
        clf = LogisticRegressionCV(Cs=Cs, cv=5, solver='liblinear')

        # fit the original training set
        clf.fit(demean_norm(X_train), y_train[:, i])

        # predict from X_test
        y_pred[:, i] = clf.predict(demean_norm(X_test))
        y_pred_prob[:, i] = clf.predict_proba(demean_norm(X_test))[
            :, 1]  # return probabilities

        # predict from X_train for haufe weights
        y_train_pred[:, i] = clf.predict(demean_norm(X_train))
        y_train_pred_prob[:, i] = clf.predict_proba(demean_norm(X_train))[:, 1]
    return y_pred, y_pred_prob, y_train_pred, y_train_pred_prob


def run_kfold_logistic(X, y, phenotype_labels, site_data, y_shuffle=False,
                       y_regress=False, y_control=None, perform_site=None,
                       k_iterations=100, model='metamatch',
                       k_splits=10, seed=None):
    '''
    Performs kfold cross validation for the metamatching project
    '''
    
    # replicability across models
    if seed is None:
        random_state = None
    else:
        random_state = 42+(seed*1052022)
    print("random state:", random_state)

    # prepare the xarray for model weights
    haufe_weights = xr.DataArray(np.zeros((X.shape[1],
                                           y.shape[1],
                                           k_iterations,
                                           k_splits,
                                           2)),
                                 dims=('edge', 
                                       'phenotype',
                                       'iteration', 
                                       'fold', 
                                       'type'),
                                 coords={'phenotype': phenotype_labels,
                                         'type': ['prediction', 'probability']})

    # prepare the xarray for predictions
    predictions = xr.DataArray(np.zeros((y.shape[0],
                                         y.shape[1],
                                         k_iterations,
                                         3)),
                               dims=('participant', 'phenotype',
                                     'iteration', 'type'),
                               coords={'phenotype': phenotype_labels,
                                       'type': ['prediction', 'probability', 'cv']})

    # prepare UKBB/ PCA weights
    if model == 'metamatch' or model == 'PCAlogreg':
        print("assigning mm weights...")
        dim_reduced_weights = xr.DataArray(np.zeros((mm_n_components,
                                                     y.shape[1],
                                                     k_iterations,
                                                     k_splits)),
                                           dims=('component', 'phenotype',
                                                 'iteration', 'fold'),
                                           coords={
                                               'phenotype': phenotype_labels})
    if model == 'multilayer_metamatch':
        dim_reduced_weights = xr.DataArray(np.zeros((ml_mm_n_components,
                                                     y.shape[1],
                                                     k_iterations,
                                                     k_splits)),
                                           dims=('component', 'phenotype',
                                                 'iteration', 'fold'),
                                           coords={
                                               'phenotype': phenotype_labels})

    else:
        dim_reduced_weights = None

    if y_shuffle:
        # shuffle based on the original data
        y_orig = y.copy()  
        site_data_orig = site_data.copy()
        if y_regress:
            y_control_orig = y_control.copy()
        
    for k_it in range(k_iterations):

        if y_shuffle:
            print('Shuffling data...')
            idx = np.random.permutation(len(y))
            y = y_orig[idx]

            # shuffle site and confound data too
            site = site_data_orig[idx]
            if y_regress:
                y_control = y_control_orig[idx, :]

        # create the cross validation strat
        # note that you have to stratify by site if you
        # want to harmonise it. Otherwise you might
        # train only two sites but want to test on three
        cv = StratifiedKFold(n_splits=k_splits, shuffle=True,
                             random_state=random_state)
        
        y_pred = np.zeros((y.shape))
        y_pred_prob = np.zeros((y.shape))
        split_order = np.zeros((y.shape))
        for i, (train_index, test_index) in enumerate(cv.split(X, site_data)):

            # define training and testing data for this iteration
            X_train = X[train_index, :].copy()
            y_train = y[train_index, :].copy()
            
            X_test = X[test_index, :].copy()
            y_test = y[test_index, :].copy()

            # copy X_train for Haufe transform later
            X_train_orig = X_train.copy()

            # site harmonisation
            if perform_site:
                print('Harmonizing site...')
                X_train, X_test = site_harmonisation(site_data[train_index],
                                                     site_data[test_index],
                                                     X_train, X_test,
                                                     y_train, y_test)
                assert np.sum(np.isnan(X_train)) == 0, "site harmon created nans"
                assert np.sum(np.isnan(X_test)) == 0, "site harmon created nans"

            # regress control variables
            if y_regress:
                print('Regressing control variables...')
                
                X_train, X_test = confound_regression(y_control[train_index],
                                                      y_control[test_index],
                                                      X_train, X_test)

            # normalise data (within subject, no leakage)
            X_train = demean_norm(X_train)
            X_test = demean_norm(X_test)

            # "downsample" data via metamatching or PCA
            if model == 'metamatch':
                # generate the UKBB mm features
                X_train = get_mm_predictions(X_train, y_train)
                X_test = get_mm_predictions(X_test, y_test)

            elif model == "multilayer_metamatch":
                # generate the ml mm features
                X_train = get_multilayer_mm_predictions(X_train, y_train)
                X_test = get_multilayer_mm_predictions(X_test, y_test)
                
            elif model == 'PCAlogreg':
                # reduce X to 67 features via PCA
                pca = PCA(n_components=mm_n_components)
                pca.fit(X_train)
                X_train = pca.transform(X_train)
                X_test = pca.transform(X_test)

            else:
                assert model == 'logreg'

            # run the "stacking" regression
            y_pred[test_index, :], y_pred_prob[
                test_index, :], y_train_pred, y_train_pred_prob = log_model(
                    X_train, y_train, X_test)
            split_order[test_index] = i

            # haufe transformation weights
            haufe_weights[:, :, k_it, i, 0] = covariance_rowwise(
                X_train_orig, y_train_pred).reshape(mm_n_edges, -1)
            haufe_weights[:, :, k_it, i, 1] = covariance_rowwise(
                X_train_orig, y_train_pred_prob).reshape(mm_n_edges, -1)

            # # get the haufe weights for the reduced components
            # if model == 'metamatch' or model == 'PCAlogreg':
            #     dim_reduced_weights[:, :, k_it, i] = covariance_rowwise(
            #         X_train, y_train_pred).reshape(mm_n_components, -1)
            
            if model == 'multilayer_metamatch':
                dim_reduced_weights[:, :, k_it, i] = covariance_rowwise(
                    X_train, y_train_pred).reshape(ml_mm_n_components, -1)
            
        # Account for the shuffling for the misclassification analysis
        predictions[:, :, k_it, 0] = y_pred
        predictions[:, :, k_it, 1] = y_pred_prob
        predictions[:, :, k_it, 2] = split_order
    return predictions, haufe_weights, dim_reduced_weights
