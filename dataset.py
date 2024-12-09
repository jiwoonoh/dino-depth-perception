from torch.utils.data import Dataset
import torch
import os

class DictDataset(Dataset):
    def __init__(self, data_dir):
        """
        Args:
            data_dir (str): Path to the directory containing .pt files.
        """
        self.data_dir = data_dir
        self.file_paths = sorted(
            [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.pt')]
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Load and return the dictionary from the .pt file.
        """
        file_path = self.file_paths[idx]
        data = torch.load(file_path)
        return data