#!/bin/bash

# parameters
cores=4

# run workflow for each dataset
snakemake --configfile CBN_OCD_Baseline.yml --cores $cores
snakemake --configfile SNUH_OCD.yml --cores $cores
snakemake --configfile MEL_OCD_ClinicalTrial.yml --cores $cores
