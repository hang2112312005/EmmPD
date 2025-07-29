import torch
from torch.nn import functional as F
import os
import h5py
def compute_patch_similarity(features, coords, grid_size=4):
    """
    Compute patch similarity within a 16×16 spatial sliding window and remove redundant patches.

    :param features: (N, D)
    :param coords: (N, 2)
    :param grid_size
    """
    patch_size = 512  # patch: 512×512
    step = grid_size * patch_size
    x_min, y_min = coords.min(dim=0)[0]
    x_max, y_max = coords.max(dim=0)[0]
    selected_indices = []
    # normalize
    features_norm = F.normalize(features, p=2, dim=-1)

    for x in range(x_min, x_max, step):
        for y in range(y_min, y_max, step):
            in_window = (coords[:, 0] >= x) & (coords[:, 0] < x + step) & \
                        (coords[:, 1] >= y) & (coords[:, 1] < y + step)
            window_indices = torch.where(in_window)[0]
            if len(window_indices) < 2:
                selected_indices.append(window_indices)
                continue
            window_features = features_norm[window_indices]
            # similarity
            similarity_matrix = torch.mm(window_features, window_features.t())
            # threshold
            threshold = similarity_matrix.mean()
            redundant = similarity_matrix.mean(dim=1) > threshold
            keep_indices = window_indices[~redundant]
            if len(keep_indices) == 0:
                keep_indices = window_indices[:1]
            selected_indices.append(keep_indices)
    return torch.cat(selected_indices)

def process_h5_files(input_folder, output_folder):
    """
    Process all .h5 files in the input_folder, remove redundant features, and save the results to the output_folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".h5"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            with h5py.File(input_path, "r") as f:
                features = torch.tensor(f["features"][:], dtype=torch.float32)
                coords = torch.tensor(f["coords"][:], dtype=torch.int32)
            selected_indices = compute_patch_similarity(features, coords, grid_size=4)
            compressed_features = features[selected_indices]
            compressed_coords = coords[selected_indices]
            with h5py.File(output_path, "w") as f_out:
                f_out.create_dataset("features", data=compressed_features.numpy())
                f_out.create_dataset("coords", data=compressed_coords.numpy())
    print(f"finish")

input_folder = "./dataset/cameylon+/h5_files"
output_folder = "./dataset/cameylon+/UNI_2D_h5_files_w_4_4"
process_h5_files(input_folder, output_folder)
print("The processing of all .h5 files has been completed.")
