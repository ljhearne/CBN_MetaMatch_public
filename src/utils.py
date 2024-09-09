import numpy as np
import pandas as pd
import glob
import os
import re


def array_to_mat(arr, N=419):
    mat = np.zeros((N, N))
    index = np.tril(np.ones(N), k=-1)
    mat[index == 1] = arr.copy()
    mat = mat+mat.T

    # make diagonal nans
    index = np.diag(np.ones(N))
    mat[index == 1] = np.nan
    return mat


def nedge_to_nnode(E):
    # convert E edges from a symmetric
    # matrix to the number of nodes
    return int((np.sqrt(8*E + 1) + 1) / 2)


def shuffle_array(arr):
    # shuffles an array within each column
    return np.apply_along_axis(np.random.permutation, axis=0, arr=arr)


class search_bids:
    '''
    This is a lightwight class that searches bids folders using regex
    It is a replacement for pybids.
    '''

    def __init__(self, fmriprep, bids=None):
        self.fmriprep = fmriprep
        self.bids = bids

    def get_subjects(self):
        # get a list of subjects in directory
        folders = glob.glob(self.fmriprep+'sub-*/')
        subjects = sorted([f.split('sub-')[-1].split('/')[0] for f in folders])
        return subjects

    def file_filter(self, wildcard_dict, subjects=None):
        # only searchs a subj dir at a time to speed things up

        # Initialize an empty list to store matching filenames
        matching_files = []

        if subjects is None:
            subjects = self.get_subjects()

        matching_files = []
        for subj in subjects:

            # build the wildcard information
            parts_to_match = ['sub-'+subj]
            for key, item in wildcard_dict.items():

                if item is None:
                    pass  # leave it out
                elif key == 'extension':
                    parts_to_match.append(item)
                else:
                    parts_to_match.append(key+'-'+item)

            # Combine the parts into a wildcard pattern that matches the filename in any order
            pattern = ".*(" + "|".join(parts_to_match) + ").*"

            # Compile the regular expression pattern
            regex = re.compile(pattern, re.IGNORECASE)

            # path to explore:
            directory_path = self.fmriprep+'sub-'+subj

            # Use os.walk() to traverse the directory recursively
            for dirpath, dirnames, filenames in os.walk(directory_path):
                for filename in filenames:
                    if regex.match(filename):
                        # Now check if the file contains all the parts
                        if all(part in filename for part in parts_to_match):
                            matching_files.append(
                                os.path.join(dirpath, filename))
        return matching_files

    def glob_phenotype_one_to_one(self, session=None):
        # search the bids phenotype directory and
        # perform one to one matching, i.e., length
        # of df = number of subjects.

        phenotype_df = pd.read_csv(
            self.bids+'participants.tsv', delimiter='\t').set_index('participant_id')

        for f in glob.glob(self.bids+'phenotype/*.tsv'):
            df = pd.read_csv(f, delimiter='\t')

            # isolate given session
            if 'session' in df.columns and session != None:
                df = df.loc[df['session'] == session]
                # drop this columns to prevent overlap with joining later
                df = df.drop('session', axis=1)

            if len(df) != len(phenotype_df):
                # this phenotype data has multiple instances of participant_id's
                # across rows.
                # this is most likely due to an error (!), or it has multiple
                # time points. An example of this in the CBN OCD dataset is the
                # Ybocs symptoms which contains 'past' and 'present' symptoms
                # which are coded as seperate rows
                print('Excluding', f.split('/')[-1], 'from phenotypes')
            else:
                phenotype_df = phenotype_df.join(
                    df.set_index('participant_id'), validate='one_to_one')
        return phenotype_df
