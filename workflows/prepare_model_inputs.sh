#!/bin/bash

## Generate ML model inputs
# parameters
CONFIG="../scripts/mm_config.json"
#DENOISE_list=("simple+gsr" "scrubbing.5+gsr")
DENOISE_list=("simple")
SPACE=fsLR #MNI152NLin2009cAsym
FDTHRESH=0.5
FDTIME=5

# Loop through the list using a for loop
for DENOISE in "${DENOISE_list[@]}"; do
    python ../scripts/generate_model_inputs.py --config $CONFIG --denoise $DENOISE --space $SPACE --fd_thresh $FDTHRESH --fd_time $FDTIME
done
