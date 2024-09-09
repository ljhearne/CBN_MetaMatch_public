'''
Parcellates nifti or cifti files.

Python implementation of wb_command -cifti-parcellate .. -method MEAN

see https://nbviewer.org/github/neurohackademy/nh2020-curriculum/blob/master/we-nibabel-markiewicz/NiBabel.ipynb

'''

import numpy as np
import nibabel as nb
import argparse
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import new_img_like

# Create parser for options
parser = argparse.ArgumentParser(
    description='''Run python based cifti-parcellate''')

# These parameters must be passed to the function
parser.add_argument('--input',
                    type=str,
                    default=None,
                    help='''input cifti dtseries.nii or volume nii.gz file''')

parser.add_argument('--parc',
                    type=str,
                    default=None,
                    help='''Cifti parc file (or text file) ,or nifti parc volume''''')

parser.add_argument('--output',
                    type=str,
                    default='output.csv',
                    help='''path to output file, e.g., output.csv''')


def surf_data_from_cifti(data, axis, surf_name):
    assert isinstance(axis, nb.cifti2.BrainModelAxis)
    for name, data_indices, model in axis.iter_structures():  # Iterates over volumetric and surface structures
        if name == surf_name:                                 # Just looking for a surface
            # Assume brainmodels axis is last, move it to front
            data = data.T[data_indices]
            # Generally 1-N, except medial wall vertices
            vtx_indices = model.vertex
            surf_data = np.zeros((vtx_indices.max() + 1,) +
                                 data.shape[1:], dtype=data.dtype)
            surf_data[vtx_indices] = data
            return surf_data
    raise ValueError(f"No structure named {surf_name}")


def volume_from_cifti(data, axis):
    assert isinstance(axis, nb.cifti2.BrainModelAxis)
    # Assume brainmodels axis is last, move it to front
    data = data.T[axis.volume_mask]
    # Which indices on this axis are for voxels?
    volmask = axis.volume_mask
    # ([x0, x1, ...], [y0, ...], [z0, ...])
    vox_indices = tuple(axis.voxel[axis.volume_mask].T)
    vol_data = np.zeros(axis.volume_shape + data.shape[1:],  # Volume + any extra dimensions
                        dtype=data.dtype)
    vol_data[vox_indices] = data                             # "Fancy indexing"
    return nb.Nifti1Image(vol_data, axis.affine)


def decompose_cifti(img):
    data = img.get_fdata(dtype=np.float32)
    brain_models = img.header.get_axis(1)  # Assume we know this
    return (volume_from_cifti(data, brain_models),
            surf_data_from_cifti(data, brain_models,
                                 "CIFTI_STRUCTURE_CORTEX_LEFT"),
            surf_data_from_cifti(data, brain_models, "CIFTI_STRUCTURE_CORTEX_RIGHT"))


def get_Schaefer419():
    # hard coded function to get Schaefer419 atlas

    # load the dlabel parcellation
    parc = '/home/lukeh/hpcworking/shared/parcellations/Tian2020MSA_v1.1/3T/Cortex-Subcortex/Schaefer2018_400Parcels_17Networks_order_Tian_Subcortex_S1.dlabel.nii'
    _, left, right = decompose_cifti(nb.load(parc))

    # load the volume we want to replace with
    volume = '/home/lukeh/projects/CBN_MetaMatch/data/atlas/subcortical_HCP_cbig_order.dlabel.nii'
    vol, _, _ = decompose_cifti(nb.load(volume))

    # reindex the left cortex by subtracting Ye's subcortical roi
    for i in range(17, 217):
        left[left == i] = i-16

    # Same for the right hemisphere
    for i in range(217, 417):
        right[right == i] = i-16

    # add 400 to volume
    vol_data = vol.get_fdata()
    vol_data[vol_data > 0] = vol_data[vol_data > 0]+400
    new_vol = new_img_like(vol, vol_data, affine=vol.affine, copy_header=True)
    return vol, left, right


def cifti_parcellate(input, parc, output):
    '''
    Parcellate a cifti file using a parc file.
    Should replicate wb_command cifti-parcellate -method MEAN
    '''

    # load the parcellation
    if parc.endswith('.dlabel.nii'):
        # proper parc detected, load
        pvol, pleft, pright = decompose_cifti(nb.load(parc))

    # custom parcellations
    else:
        # get Schaefer 419
        print('Creating Schaefer 419 from components')
        pvol, pleft, pright = get_Schaefer419()

    # load the timeseries (dtseries data)
    vol, left, right = decompose_cifti(nb.load(input))

    # load and flatten the volume arrays
    pvol = pvol.get_fdata().reshape(-1)
    vol = vol.get_fdata().reshape(-1, vol.shape[3])
    assert left.shape[0] == pleft.shape[0], "input and parc dims do not match!"

    # get all parcel ids across volume and surface
    parcel_ids = np.hstack(
        (np.unique(pleft), np.unique(pright), np.unique(pvol)))

    # remove '0' parcel
    parcel_ids = np.delete(parcel_ids, np.where(parcel_ids == 0)[0])

    # preallocate
    time_series = np.zeros((left.shape[1], len(parcel_ids)))

    # for each data, loop through and calculate mean.
    # loop through parcels, index unique parcel in space, avg,
    # take bold values and store in time_series
    _ids = np.unique(pleft)
    _ids = np.delete(_ids, np.where(_ids == 0)[0])
    for p in _ids:
        #p-1 for pythonic indexing
        parcel_index = np.ravel(pleft) == p
        time_series[:, int(p)-1] = np.mean(left[parcel_index, :], axis=0)

    _ids = np.unique(pright)
    _ids = np.delete(_ids, np.where(_ids == 0)[0])
    for p in _ids:
        #p-1 for pythonic indexing
        parcel_index = np.ravel(pright) == p
        time_series[:, int(p)-1] = np.mean(right[parcel_index, :], axis=0)

    _ids = np.unique(pvol)
    _ids = np.delete(_ids, np.where(_ids == 0)[0])
    for p in _ids:
        #p-1 for pythonic indexing
        parcel_index = pvol == p
        time_series[:, int(p)-1] = np.mean(vol[parcel_index], axis=0)

    # save out
    np.savetxt(output, time_series, delimiter=',')
    return time_series


def nifti_parcellate(input, parc, output):
    masker = NiftiLabelsMasker(labels_img=parc, memory='../data/scratch/', memory_level=5, verbose=1)
    time_series = masker.fit_transform(input)

    # save out
    np.savetxt(output, time_series, delimiter=',')
    return time_series

if __name__ == '__main__':
    # Read in user-specified parameters
    args = parser.parse_args()
    
    # run parcellate
    if '.nii.gz' in args.input:
        print('NIFTI assumed')
        nifti_parcellate(args.input, args.parc, args.output)
    elif '.dtseries.nii' in args.input:
        print('CIFTI assumed')
        cifti_parcellate(args.input, args.parc, args.output)
