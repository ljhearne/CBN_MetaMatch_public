
# this code takes the metamatching model defined and resaves it
# so that it can be reloaded more simply. This only needs to be
# run a single time

import os
import sys
import torch


if __name__ == '__main__':
    # paths
    path_repo = '../data/Meta_matching_models-main/'
    path_v11 = os.path.join(path_repo, 'v1.1')
    path_v10 = os.path.join(path_repo, 'v1.0')

    # device (may be important if not running on local cluster)
    gpu = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sys.path.append(path_v10)
    import CBIG_model_pytorch as CBIG_model_pytorch

    # model v11
    path_model_weight = os.path.join(
        path_v11, 'meta_matching_v1.1_model.pkl_torch')
    net = torch.load(path_model_weight, map_location=device)
    model_scripted = torch.jit.script(net)  # Export to TorchScript
    path_out = os.path.join(path_v11, 'meta_matching_v1.1_model_scripted.pt') 
    model_scripted.save(path_out)  # Save
