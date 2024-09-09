"""
Generates subject lists based on the amount of
time in a scan with low motion based on framewise 
displacement
"""

import pandas as pd
import numpy as np
import json
from src.utils import search_bids


def generate_subject_lists(fmriprep, space=None, session=None, group=None, fd_thresh=0.5, fd_time=5):
    '''
    Generates a list of subjects that satisify the FD threshold and FD time
    threshold based on a BIDSlayout.
    Includes an extra "session" parameter to deal with different datasets in
    the metamatch project.
    '''
    layout = search_bids(fmriprep)
    subject_list = layout.get_subjects()
    new_subject_list = []
    mean_fd_list = []

    # if applicable, include by group
    if group != None:
        idx = np.zeros((len(subject_list)))
        for i, s in enumerate(subject_list):
            if group in s:
                idx[i] = 1
            else:
                idx[i] = 0
        subject_list = list(np.array(subject_list)[idx == 1])

    for subj in sorted(subject_list):

        # get file
        f = layout.file_filter({'task': 'rest',
                                'ses': session,
                                'desc': 'confounds',
                                'extension': 'tsv'},
                               subjects=[subj])
        assert len(f) == 1, "Error in fetching confound file..."

        # get the average framewise displacement
        # (note this is prior to any scrubbing so
        # high motion volumes are included in the mean)
        fd = pd.read_csv(f[0], delimiter="\t")['framewise_displacement'].values
        mean_fd = np.nanmean(fd)

        # calculate how much data will be kept based on fd
        # get TR
        f = layout.file_filter({'task': 'rest',
                                'ses': session,
                                'space': 'MNI152NLin2009cAsym',
                                'desc': 'preproc',
                                'extension': 'json'},
                               subjects=[subj])
        # assert a json file is found
        #(it's okay if there is two, we are just getting the TR)
        assert len(f) != 0, "Error in fetching json file..."

        tr = json.load(open(f[0],))['RepetitionTime']
        remaining_time = (sum(fd < fd_thresh) * tr)

        if mean_fd < fd_thresh and remaining_time > fd_time * 60:
            new_subject_list.append(subj)
            mean_fd_list.append(mean_fd)
        else:
            print(subj, "excluded due to FD")
    label = 'FD-'+str(int(fd_thresh*100))+'_N-'+str(len(new_subject_list))

    return new_subject_list, label, mean_fd_list
