import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class MRIDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the .pt files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = []
        self.labels = []
        
        # Traverse through the directory and collect all file paths and labels
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    if file_name.endswith('.pt'):
                        self.file_paths.append(os.path.join(class_dir, file_name))
                        self.labels.append(class_name)  # Assuming folder names are labels
        
        # Convert labels to indices
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(set(self.labels)))}
        self.labels = [self.class_to_idx[label] for label in self.labels]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        series = torch.load(file_path)  # Load the .pt file (expected shape: [depth, H, W])

        # Ensure series is of shape [depth, H, W], then unsqueeze to [1, depth, H, W] to add channel dimension
        # if series.dim() == 3:
            # series = series.unsqueeze(0)  # Add channel dimension at position 0, resulting in shape: [1, depth, H, W]
        
        # Check the shape to ensure it's correct
        # assert series.shape[0] == 1, f"Expected 1 channel, but got {series.shape[0]} channels"
        series = series.permute(1, 0, 2, 3)
        label = self.labels[idx]
        
        if self.transform:
            series = self.transform(series)
        
        return series, label


class MRIDataset2D(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the .pt files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = []
        self.labels = []
        
        # Traverse through the directory and collect all file paths and labels
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    if file_name.endswith('.pt'):
                        self.file_paths.append(os.path.join(class_dir, file_name))
                        self.labels.append(class_name)  # Assuming folder names are labels
        
        # Convert labels to indices
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(set(self.labels)))}
        self.labels = [self.class_to_idx[label] for label in self.labels]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        series = torch.load(file_path)  # Load the .pt file (expected shape: [depth, H, W])
        
        # Apply Maximum Intensity Projection (MIP) to get a 2D image
        mip_image = torch.max(series, dim=0)[0]  # MIP along the depth dimension

        # Convert MIP image to 3D shape expected by a 2D CNN (1, H, W)
        mip_image = mip_image.unsqueeze(0)  # Add channel dimension at position 0

        label = self.labels[idx]
        
        if self.transform:
            mip_image = self.transform(mip_image)
        
        return mip_image, label


class MRIDataset2DMiddleSlice(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the .pt files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = []
        self.labels = []
        
        # Traverse through the directory and collect all file paths and labels
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    if file_name.endswith('.pt'):
                        self.file_paths.append(os.path.join(class_dir, file_name))
                        self.labels.append(class_name)  # Assuming folder names are labels
        
        # Convert labels to indices
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(set(self.labels)))}
        self.labels = [self.class_to_idx[label] for label in self.labels]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        series = torch.load(file_path)  # Load the .pt file (expected shape: [depth, H, W])
        
        # Apply Maximum Intensity Projection (MIP) to get a 2D image
        mip_image = torch.max(series, dim=0)[0]  # MIP along the depth dimension

        # Convert MIP image to 3D shape expected by a 2D CNN (1, H, W)
        mip_image = mip_image.unsqueeze(0)  # Add channel dimension at position 0

        label = self.labels[idx]
        
        if self.transform:
            mip_image = self.transform(mip_image)
        
        return mip_image, label