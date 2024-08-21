import os
import torch
from torch.utils.data import Dataset
import numpy as np

def get_files(dataroot):
    files_list = []
    for root, dirs, files in os.walk(dataroot):
        for file in files:
            if file.endswith('.npy'):
                files_list.append(os.path.join(root, file))
    return files_list

class DAVISDataset(Dataset):
    def __init__(
        self, root_dir):

        self.root_dir = root_dir
        self.video_paths = get_files(self.root_dir)
        self.video_paths = sorted(self.video_paths)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        loaded_numpy_array = np.load(video_path)
        tensor = torch.tensor(loaded_numpy_array)
        return tensor