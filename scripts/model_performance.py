# %%
'''
05/02/2024
A note - trying to debug this code but the input data has changed shape (the .csvs),
thus all the models need to be rerun. (From memory I have included 4 more people because
they were only missing data that wasn't important).

No point rerunning the models until we freshly preprocess the melbourne data.

'''
from src.data_loader import result_loader
import xarray as xr
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from mlconfound.stats import partial_confound_test
from src.data_loader import load_model_inputs


def load_confounds(confound_labels=['gender', 'age',
                                    'avg_framewise_displacement',
                                    'handedness', 'site'],
                   ds_list=["MEL_OCD_ClinicalTrial",
                            "SNUH_OCD",
                            "SNUH_HC",
                            "CBN_OCD",
                            "CBN_HC"]):
    # gather confound data

    # has no bearing on demographics
    denoise = 'simple+gsr'

    # Get model inputs
    y_df, _ = load_model_inputs(ds_list, denoise)

    # ml for handedness
    y_df['handedness'].replace({'right': 0, 'left': 1}, inplace=True)

    return y_df[confound_labels]


def evaluate_log_predictions(y_test, y_pred, y_pred_prob, labels):
    # compares real and predicted scores with a few
    # different metrics

    results_df = pd.DataFrame()
    roc_df = pd.DataFrame()

    for phenotype in range(y_test.shape[1]):

        # sensitivity / specificity
        tn, fp, fn, tp = confusion_matrix(y_test[:, phenotype],
                                          y_pred[:, phenotype]).ravel()
        sensitivity = tp / (tp+fn)
        specificity = tn / (tn+fp)

        # calculate balanced accuracy
        acc = balanced_accuracy_score(
            y_test[:, phenotype], y_pred[:, phenotype])

        # roc auc
        roc_auc = roc_auc_score(
            y_test[:, phenotype], y_pred_prob[:, phenotype])

        # save per-fold results
        df = pd.DataFrame.from_dict({'phenotype': labels[phenotype],
                                     'acc': [acc],
                                     'roc_auc': [roc_auc],
                                     'sensitivity': [sensitivity],
                                     'specificity': [specificity]
                                     })
        results_df = pd.concat([results_df, df])

        # calculate roc curve
        fpr, tpr, thresholds = roc_curve(
            y_test[:, phenotype], y_pred_prob[:, phenotype])

        # save roc curve results
        df = pd.DataFrame.from_dict({'phenotype': labels[phenotype],
                                     'fpr': fpr,
                                     'tpr': tpr,
                                     'thresholds': thresholds
                                     })
        roc_df = pd.concat([roc_df, df])

    return results_df, roc_df


def isinteger(x):
    return np.equal(np.mod(x, 1), 0)


def confounder_test(confounds, y, y_pred_prob, num_perms=10):

    # wrapper for confounder test
    results = pd.DataFrame()
    for (colname, colval) in confounds.items():
        H0_y = y.reshape(-1)
        H0_yhat = y_pred_prob.reshape(-1)
        H0_c = colval.values.reshape(-1)

        # if the data are int and under 5,
        # test as a cat
        if all(isinteger(H0_c)) and np.max(H0_c) < 5:
            cat_c = True
        else:
            cat_c = False

        res = partial_confound_test(H0_y, H0_yhat, H0_c,
                                    cat_y=True, cat_yhat=False,
                                    cat_c=cat_c,
                                    num_perms=num_perms,
                                    return_null_dist=True)
        df1 = pd.DataFrame({
            'confound': [colname],
            'r2_yhat_c': res.r2_yhat_c,
            'p': res.p,
            'permuted': False
        })

        df2 = pd.DataFrame({
            'r2_yhat_c': res.null_distribution,
            'confound': colname,
            'permuted': True
        })
        results = pd.concat([results, df1, df2], ignore_index=True)

    return results


def calculate_performance(model, path_to_results, phenotype, denoise, n_batch,
                          ksplits, n_batch_permutations, n_confound_perms,
                          resample):
    print(model)

    # models to loop through
    model_suffix_list = ['', '_site_confounds_removed']

    # Load confound data
    confounds = load_confounds()

    # Init. result dfs
    accuracy = []
    roc = []
    confounders = []

    # Load inputs (y)
    in_file = (f"{path_to_results}/{model}/"
               f"kfold_denoise-{denoise}_ksplit-{ksplits}_"
               f"model-{model}_batch-0_{phenotype}_")
    if resample != 0:
        in_file = f"{in_file}N{resample}_"

    y = xr.load_dataset(f"{in_file}inputs.nc")['y'].to_numpy()

    # Iterate through the model types
    for model_suffix in model_suffix_list:
        print(model_suffix)

        # load predictions
        predictions = result_loader(model=model,
                                    model_suffix=model_suffix,
                                    phenotype=phenotype,
                                    denoise=denoise,
                                    ksplit=ksplits,
                                    output_type='predictions',
                                    n_batch=n_batch,
                                    resample=resample,
                                    path=path_to_results
                                    ).to_array().to_numpy()

        # Iterate through iterations of the kfold
        for i in range(predictions.shape[3]):

            # Get predictions
            y_pred = predictions[0, :, 0, i, 0].reshape(-1, 1)
            y_pred_prob = predictions[0, :, 0, i, 1].reshape(-1, 1)
            splits = predictions[0, :, 0, i, 2].reshape(-1, 1)

            # Iterate through folds
            for fold in range(ksplits):

                # index the appropriate test split
                idx = (splits == 0).reshape(-1)

                # Calculate and save model accuracy
                df1, df2 = evaluate_log_predictions(y[idx, :],
                                                    y_pred[idx, :],
                                                    y_pred_prob[idx, :],
                                                    [phenotype]*len(y[idx]))
                data = {
                    'model': model + model_suffix,
                    'denoise': denoise,
                    'permuted': False,
                    'iteration': i,
                    'fold': fold
                }

                df1 = df1.assign(**data)
                df2 = df2.assign(**data)
                accuracy.append(df1)
                roc.append(df2)

                # Calculate Spisak's mlconfound
                if n_confound_perms > 0:

                    df3 = confounder_test(confounds, y, y_pred_prob,
                                          num_perms=n_confound_perms)
                    data = {
                        'model': model + model_suffix,
                        'denoise': denoise,
                        'iteration': i,
                        'fold': fold}
                    df3 = df3.assign(**data)
                    confounders.append(df3)

        # Calculate null permutations
        if n_batch_permutations > 0:
            print('\tCalculating nulls...')
            # load relevant shuffled permutations
            predictions = result_loader(model=model,
                                        model_suffix=model_suffix,
                                        phenotype=phenotype,
                                        denoise=denoise,
                                        ksplit=ksplits,
                                        output_type='predictions',
                                        n_batch=n_batch_permutations,
                                        resample=resample,
                                        shuffled=True,
                                        path=path_to_results
                                        ).to_array().to_numpy()

            # Iterate through shuffled predictions

            for i in range(predictions.shape[3]):

                # Get predictions
                y_pred = predictions[0, :, 0, i, 0].reshape(-1, 1)
                y_pred_prob = predictions[0, :, 0, i, 1].reshape(-1, 1)
                splits = predictions[0, :, 0, i, 2].reshape(-1, 1)

                # Iterate through folds
                for fold in range(ksplits):

                    # index the appropriate test split
                    idx = (splits == 0).reshape(-1)

                    # Calculate and save model accuracy
                    df1, df2 = evaluate_log_predictions(y[idx, :],
                                                        y_pred[idx, :],
                                                        y_pred_prob[idx, :],
                                                        [phenotype]*len(y[idx]))
                    data = {
                        'model': model + model_suffix,
                        'denoise': denoise,
                        'permuted': True,
                        'iteration': i,
                        'fold': fold
                    }

                    df1 = df1.assign(**data)
                    df2 = df2.assign(**data)
                    accuracy.append(df1)
                    roc.append(df2)

    # Concatenate DataFrames outside the loop and save out
    out_path = '../results/model_performance/'
    if resample != 0:
        out_file = f"model-{model}_phenotype-{phenotype}_denoise-{denoise}_N{resample}.csv"
    else:
        out_file = f"model-{model}_phenotype-{phenotype}_denoise-{denoise}.csv"

    accuracy_df = pd.concat(accuracy, ignore_index=True)
    roc_df = pd.concat(roc, ignore_index=True)

    accuracy_df.to_csv(out_path+'accuracy_'+out_file,
                       index=False)

    roc_df.to_csv(out_path+'roc_'+out_file,
                  index=False)

    if n_confound_perms > 0:
        confounders_df = pd.concat(confounders, ignore_index=True)
        confounders_df.to_csv(out_path+'confounders_'+out_file,
                              index=False)


if __name__ == "__main__":
    # PARAMETERS
    path_to_results = '/home/lukeh/hpcworking/lukeH/projects/CBN_MetaMatch/results/kfold_july/'
    phenotype = 'group'

    # Main results with confounds and permutations:
    # (very slow with confound modelling)
    denoise = "simple+gsr"
    for model in ["multilayer_metamatch", "logreg"]:
        calculate_performance(model, path_to_results, phenotype, denoise,
                              n_batch=200, ksplits=5, n_batch_permutations=1000,
                              n_confound_perms=10, resample=0)

    # for model in ["metamatch"]:
    #     calculate_performance(model, path_to_results, phenotype, denoise,
    #                           n_batch=200, ksplits=5, n_batch_permutations=0,
    #                           n_confound_perms=0, resample=0)

    # Resampling results:
    for model in ["multilayer_metamatch", "logreg"]:
        for resample in [100, 150, 200, 250, 300]:
            calculate_performance(model, path_to_results, phenotype, denoise,
                                  n_batch=200, ksplits=5, n_batch_permutations=0,
                                  n_confound_perms=0, resample=resample)

    # # Alternative denoising results:
    # denoise = "simple"
    # for model in ["multilayer_metamatch", "logreg", "metamatch"]:
    #     calculate_performance(model, path_to_results, phenotype, denoise,
    #                           n_batch=200, ksplits=5, n_batch_permutations=0,
    #                           n_confound_perms=0, resample=0)
