"""
Post-fmriprep denoising using Nilearn.

Uses high-level nilearn functions to clean timeseries data.

This approach and the possible strategies are from this paper:
see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10153168/pdf/nihpp-2023.04.18.537240v3.pdf

The specific 'out-of-the-box' denoise strategies are found here:
https://github.com/SIMEXP/fmriprep-denoise-benchmark/blob/b9d44504384b3641dbd1d063105cb6eb99713488/fmriprep_denoise/dataset/benchmark_strategies.json#L4

"""
import json
from nilearn.signal import clean
import numpy as np
import argparse
from nilearn.interfaces.fmriprep import load_confounds_strategy


# Create parser for options
parser = argparse.ArgumentParser(
    description='''Run nilearn based BOLD denoising''')

# These parameters must be passed to the function
parser.add_argument('--input',
                    type=str,
                    default=None,
                    help='''input bold data as csv''')

parser.add_argument('--strategy',
                    type=str,
                    default=None,
                    help='''denoise strategy located in denoise_strategies.json''')

parser.add_argument('--ref_img',
                    type=str,
                    default=None,
                    help='''fmriprep reference image''')

parser.add_argument('--output',
                    type=str,
                    default=None,
                    help='''output file''')


def denoise_timeseries(input, strategy, ref_img, output):

    # interpret the denoise strategy based on the json
    # load denoise dict (assumed to be in same location)
    parameters = json.load(open('../src/denoise_strategies.json',))[strategy]

    # get confounds
    confounds, sample_mask = load_confounds_strategy(ref_img, **parameters)

    # get timeseries
    timeseries = np.loadtxt(input, delimiter=',')

    # clean the timeseries
    # note filtering is already done on the confounds
    clean_timeseries = clean(
        timeseries,
        detrend=True,
        standardize=True,
        sample_mask=sample_mask,
        confounds=confounds
        )

    # save out
    np.savetxt(output, clean_timeseries, delimiter=',')
    return None


if __name__ == '__main__':
    # Read in user-specified parameters
    args = parser.parse_args()

    # run denoise
    denoise_timeseries(args.input, args.strategy, args.ref_img, args.output)
