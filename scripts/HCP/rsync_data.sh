#!/bin/bash

# init subject list
subject_list=$(<292_unrelated_participants.txt)

# paths
path_to_rest_data="/home/lukeh/LabData/Lab_MichaelB/HumanConnectomeProject/HCP_Data_900_Subjects/Data/MRI Preprocessed/Resting State FIX-Denoised (Compact)/"

# rsync to ldrive
dest="/home/lukeh/LabData/Lab_LucaC/HumanConnectomeProject/rawdata/"

counter=0  # Initialize counter
for subj in $subject_list; do
    counter=$((counter + 1))  # Increment counter
    echo "Iteration: $counter, Subject: $subj"
    source="${path_to_rest_data}${subj}_3T_rfMRI_REST_fix.zip"
    rsync "$source" "$dest" --ignore-existing
done