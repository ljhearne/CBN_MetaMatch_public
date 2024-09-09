#!/bin/bash

# Unzips the raw HCP data
data_types=("3T_rfMRI_REST_fix")
readarray -t subj_array < 292_unrelated_participants.txt

input=/home/lukeh/LabData/Lab_LucaC/HumanConnectomeProject/rawdata/
dest=/home/lukeh/LabData/Lab_LucaC/HumanConnectomeProject/unzipped_rawdata/

count=0
for subj in "${subj_array[@]}"; do
    counter=$((counter + 1))  # Increment counter
    echo "Iteration: $counter, Subject: $subj"
    for data in "${data_types[@]}"; do
        unzip -n "${input}${subj}_${data}.zip" -d ${dest}${data}
    done
done
