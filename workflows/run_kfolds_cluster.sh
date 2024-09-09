#!/bin/bash

DENOISE_list=("scrubbing.5+gsr" "simple+gsr")
#PHENOTYPE_list=("group" "gender")
PHENOTYPE_list=("group")
N_ITS=100
N_PERMS=1000

# Loop through the list using a for loop
for PHEN in "${PHENOTYPE_list[@]}"; do
    echo $PHEN
    for DENOISE in "${DENOISE_list[@]}"; do
        echo $DENOISE

        # Run actual models (different parameters for models)
        # qsub -l mem=6gb,walltime=0:20:00,ncpus=3,chip=Intel \
        # -J 0-$N_ITS -v DENOISE="$DENOISE",MODEL="metamatch",PHENOTYPE_LABELS="$PHEN",CONFOUNDS="age gender",PERMUTE="False",RESAMPLE="0" \
        # run_kfold.pbs

        qsub -l mem=6gb,walltime=1:30:00,ncpus=3,chip=Intel \
        -J 0-$N_ITS -v DENOISE="$DENOISE",MODEL="logreg",PHENOTYPE_LABELS="$PHEN",CONFOUNDS="age gender",PERMUTE="False",RESAMPLE="0" \
        run_kfold.pbs

        # Run permutations
        # qsub -l mem=6gb,walltime=0:20:00,ncpus=3,chip=Intel \
        # -J 0-$N_PERMS -v DENOISE="$DENOISE",MODEL="metamatch",PHENOTYPE_LABELS="$PHEN",CONFOUNDS="age gender",PERMUTE="True",RESAMPLE="0" \
        # run_kfold.pbs

        qsub -l mem=6gb,walltime=1:30:00,ncpus=3,chip=Intel \
        -J 0-$N_PERMS -v DENOISE="$DENOISE",MODEL="logreg",PHENOTYPE_LABELS="$PHEN",CONFOUNDS="age gender",PERMUTE="True",RESAMPLE="0" \
        run_kfold.pbs

    done
done
