# %%
# this only needs to be run once.
import numpy as np
import os
import random

# init subject lists
n = 292  # 292 is all the data available
subject_list350 = np.loadtxt('/home/lukeh/LabData/Lab_LucaC/HumanConnectomeProject/350_unrelated_participants.txt')
new_subject_list = []
# paths
path_to_rest_data="/home/lukeh/LabData/Lab_MichaelB/HumanConnectomeProject/HCP_Data_900_Subjects/Data/MRI Preprocessed/Resting State FIX-Denoised (Compact)/"

# check that the data exists and create new subject list
for subj in subject_list350:
    subj = int(subj)
    file = path_to_rest_data+str(subj)+"_3T_rfMRI_REST_fix.zip"
    if os.path.isfile(file):
        new_subject_list.append(subj)

# shuffle the list
random.shuffle(new_subject_list)
subject_list = sorted(new_subject_list[0:n])

# save
with open(str(n)+'_unrelated_participants.txt', 'w') as f:
    for line in subject_list:
        f.write(f"{line}\n")

# %%
