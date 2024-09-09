
# %%
import numpy as np
from scipy.stats import ttest_1samp
from tqdm import tqdm
import pickle
# see https://stackoverflow.com/questions/3685265/how-to-write-a-multidimensional-array-to-a-text-file

# parameters
in_data = "/home/lukeh/LabData/Lab_LucaC/HumanConnectomeProject/derivatives/fc/"
label_file = "/home/lukeh/LabData/Lab_LucaC/Luke/Backups/hpc_backups/parcellations/Tian2020MSA_2023/3T/Cortex-Subcortex/Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S4_label.txt"
n_rois = 454

# NAc-shell-rh/lh, PUT-VA-rh/lh, CAU-DA-rh/lh, PUT-DP-rh/lh
#tian_indices = np.array([[23, 50],
#                          [17, 44],
#                          [12, 39],
#                          [15, 42]])
seeds = [23, 50,
         17, 44,
         12, 39,
         15, 42]
# 

parcel_labels = []
with open(label_file, 'r') as f:
    for count, line in enumerate(f, start=0):
        if count % 2 == 0:
            parcel_labels.append(line.split('\n')[0])

# load subject
subject_list = np.loadtxt('287_unrelated_participants.txt', dtype=str)
subject_list = subject_list[0:250]

# get fc
fc_mats = np.empty((n_rois, n_rois, len(subject_list)))
for i, sid in tqdm(enumerate(subject_list)):
    file = f"{in_data}sub-{sid}/sub-{sid}_task-rest_run-all_dir-all_parc-Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S4_method-correlation.csv"
    fc = np.loadtxt(file, delimiter=',')
    fc[np.diag_indices_from(fc)] = np.nan  # Explicitly assign nan to self connections
    fc_mats[:, :, i] = fc.copy()

output = open('../../data/HCP_fc/fc_mats.pkl', 'wb')
pickle.dump(fc_mats, output)
output.close()

# %%


# %% [markdown]
# # Seeds
# ## Ye's parcel ids:
# - NAc-shell-rh/lh (right hemisphere / left hemisphere)
# - PUT-VA-rh/lh
# - CAU-DA-rh/lh
# - PUT-DP-rh/lh
# 
# ## Corresponds to Seb's:
# - NACC
# - Ventral Putamen
# - Dorsal Caudate
# - Dorsal Putamen
# 

# # %%
# for seed in seeds:
#     # get seed-specific data
#     data = fc_mats[seed, :, :].T

#     # perform t-test
#     result = ttest_1samp(data, popmean=0, alternative='greater')

#     # save the cortical (400) t vals
#     tvals = result.statistic[54::]
#     np.savetxt(f"../../results/HCP_roi/{parcel_labels[seed]}_{seed}_fc_tvals_parc-order.csv", 
#                tvals, delimiter=',')

#     # sort the t_values high to low
#     # note that subcortical data is removed: we only want cortical pathways
#     # this reindexes the parcels but this is OK
#     # because in the metamatching parcellation it starts with the
#     # cortical data
#     sorted_tvals = np.argsort(result.statistic[54::])[::-1]
#     print(sorted_tvals[0:10])
#     # save the ordered matrix so that N rois can be grabbed in future
#     np.savetxt(f"../../results/HCP_roi/{parcel_labels[seed]}_{seed}_fc_tvals_sorted.csv",
#                sorted_tvals, delimiter=',')

#     # plot the untresholded data
#     data = np.mean(data.T, axis=1)
#     #quick_surf_plot(data, parc_file, title=parcel_labels[seed]+' FC')

#     # plot thresholded data
#     index = sorted_tvals[0:10] + 54
#     data[index] = data[index]*10
#     #quick_surf_plot(data, parc_file, parcel_labels[seed]+' Thresh')

# # %%
# break

# # %%
# # publication plot

# # %%
# import nibabel as nb
# import hcp_utils as hcp
# import nilearn.plotting as plotting
# import matplotlib.pyplot as plt
# import numpy as np

# parc_file = "/home/lukeh/LabData/Lab_LucaC/Luke/Backups/hpc_backups/parcellations/Tian2020MSA_2023/3T/Cortex-Subcortex/Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S4.dlabel.nii"


# def pub_surf_plot(input_array, input_array2, parcellation, save=None):
#     vmax = np.max(abs(input_array))
#     # a very quick nilearn plot
#     parc = np.squeeze(nb.load(parcellation).get_fdata())
#     output = np.zeros((parc.shape))
#     output2 = np.zeros((parc.shape))
#     parcels = np.unique(parc)
#     parcels = np.delete(parcels, 0)  # assume 0 denotes no roi

#     for i in parcels:
#         index = (parc==i)
#         output[index] = input_array[int(i)-1]
#         output2[index] = input_array2[int(i)-1]

#     fig = plt.figure(figsize=(3, 3))
#     count = 1
#     for view in ['lateral', 'medial']:
#         for hemi in ['left', 'right']:
        
#             if hemi == 'left':
#                 surface = hcp.left_cortex_data(output)
#                 surface2 = hcp.left_cortex_data(output2)
#                 mesh = hcp.mesh.inflated_left
#                 bg_map = hcp.mesh.sulc_left

#             elif hemi == 'right':
#                 surface = hcp.right_cortex_data(output)
#                 surface2 = hcp.right_cortex_data(output2)
#                 mesh = hcp.mesh.inflated_right
#                 bg_map = hcp.mesh.sulc_right

#             ax = fig.add_subplot(2, 2, count, projection='3d')
#             plotting.plot_surf_stat_map(mesh, surface, bg_map=bg_map, view=view, cmap='coolwarm',
#                                         hemi=hemi, bg_on_data=True, darkness=0.75,
#                                         axes=ax, alpha=1, vmax=vmax, colorbar=False)
#             # for roi in np.unique(surface2)[1::]:
#             #     try:
#             #         plotting.plot_surf_contours(mesh, surface2, levels=[int(roi)], bg_map=bg_map, view=view, colors='k', hemi=hemi, bg_on_data=True, darkness=0.50, axes=ax)
#             #     except:
#             #         print("contour not plotting", hemi, view)
#             count = count+1
#     #plt.savefig('test.svg')
#     fig.subplots_adjust(wspace=None, hspace=None)
#     if save is not None:
#         plt.savefig(save)
#     plt.show()

# # %%
# n_targets = 10
# # recall
# seeds = [23, 50,
#          12, 39,
#          17, 44,
#          15, 42]

# # parameters
# in_data = "/home/lukeh/LabData/Lab_LucaC/HumanConnectomeProject/derivatives/fc/"
# label_file = "/home/lukeh/LabData/Lab_LucaC/Luke/Backups/hpc_backups/parcellations/Tian2020MSA_2023/3T/Cortex-Subcortex/Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S4_label.txt"
# n_rois = 454

# # NAc-shell-rh/lh, PUT-VA-rh/lh, CAU-DA-rh/lh, PUT-DP-rh/lh
# seeds = [23, 50,
#          12, 39,
#          17, 44,
#          15, 42]

# parcel_labels = []
# with open(label_file, 'r') as f:
#     for count, line in enumerate(f, start=0):
#         if count % 2 == 0:
#             parcel_labels.append(line.split('\n')[0])

# for seed_a, seed_b in zip([23, 12, 17, 15], [50, 39, 44, 42]):
#     # get seed-specific data
#     seed_a_data = np.loadtxt(f"../../results/HCP_roi/{parcel_labels[seed_a]}_{seed_a}_fc_tvals_parc-order.csv", delimiter=',')
#     seed_b_data = np.loadtxt(f"../../results/HCP_roi/{parcel_labels[seed_b]}_{seed_b}_fc_tvals_parc-order.csv", delimiter=',')

#     # average the data across hemisphere seeds
#     # (too many plots otherwise)
#     avg_data = np.mean(np.vstack((seed_a_data, seed_b_data)), axis=0)

#     # plot the unthresholded data
#     plot_data = np.zeros((454))
#     plot_data[54::] = avg_data.copy()

#     # plot thresholded data
#     plot_data_threshold = np.zeros((454))
#     file_a = f"../../results/HCP_roi/{parcel_labels[seed_a]}_{seed_a}_fc_tvals_sorted.csv"
#     file_b = f"../../results/HCP_roi/{parcel_labels[seed_b]}_{seed_b}_fc_tvals_sorted.csv"
#     rois = np.stack((np.loadtxt(file_a, delimiter=',')[0:n_targets].astype(int),
#             np.loadtxt(file_b, delimiter=',')[0:n_targets].astype(int))).flatten()

#     plot_data_threshold[rois+54] = rois+54
#     mask = np.zeros((454))
#     mask[rois+54] = 1

#     plot_data[mask==0] = 0
#     pub_surf_plot(plot_data, plot_data_threshold, parc_file)

# # %%



