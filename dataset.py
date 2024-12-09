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

class RepeatedDictDataset(Dataset):
    def __init__(self, file_path, dataset_size):
        """
        Args:
            file_path (str): Path to the .pt file.
            dataset_size (int): Number of times to repeat the file.
        """
        self.file_path = file_path
        self.dataset_size = dataset_size

    def __len__(self):
        # The dataset has a fixed size
        return self.dataset_size

    def __getitem__(self, idx):
        """
        Load and return the dictionary from the .pt file.
        Repeats the same file regardless of the index.
        """
        data = torch.load(self.file_path)
        return data