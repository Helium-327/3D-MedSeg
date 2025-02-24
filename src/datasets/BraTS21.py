# -*- coding: UTF-8 -*-
'''
Description:         BraTS数据集类, 用于加载数据集, 切分数据集

Created on:         2024/07/23 16:18:28
Author:             @ Mr_Robot
Current State:      To be Refine
'''

import os
import time
from typing import List, Tuple, Optional

import pandas as pd
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Define constants for modality names
MODALITIES = ("t1", "t1ce", "flair", "t2", "seg")

class BraTS21_3D(Dataset):
    """
    A dataset class for loading BraTS 3D MRI data.
    """
    def __init__(self, data_file: str, transform: Optional[callable] = None, 
                 local_train: bool = False, length: Optional[int] = None):
        """
        Initialize the BraTS dataset.
        
        :param data_file: Path to the dataset file (CSV or TXT).
        :param transform: Optional transform to be applied to the data.
        :param local_train: Whether to load only a subset of the dataset (default: False).
        :param length: Number of samples to load (default: None, loads all).
        """
        self.data_file = data_file
        self.local_train = local_train
        self.length = length
        self.transform = transform
        self.patients_dirs, self.patients_ids = self.load_patient()
        
    def __len__(self) -> int:
        """
        Return the number of patients in the dataset.
        """
        return len(self.patients_ids)
    
    def load_patient(self, data_file: Optional[str] = None) -> Tuple[List[str], List[str]]:
        """
        Load patient directories and IDs from the dataset file.
        
        :param data_file: Path to the dataset file (optional).
        :return: Tuple of (patient directories, patient IDs).
        """
        if data_file:
            df = pd.read_csv(data_file)
        else:
            df = pd.read_csv(self.data_file)

        if self.local_train:
            df = df.iloc[:self.length]  # Load a subset for local training

        patients_dirs = df['patient_dir'].tolist()
        patients_ids = df['patient_idx'].tolist()
        
        return patients_dirs, patients_ids
    
    def load_image(self, patient_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load all modalities of a patient's MRI data.
        
        :param patient_dir: Path to the patient's directory.
        :return: Tuple of (multi-modal image tensor, mask tensor).
        """
        patient_id = os.path.basename(patient_dir)
        image_paths = [os.path.join(patient_dir, f"{patient_id}_{modality}.nii.gz") for modality in MODALITIES]
        
        # Check if all required files exist
        for path in image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} does not exist")
        
        # Load and transpose the data
        image_data = [np.transpose(nib.load(path).get_fdata(), (2, 0, 1)) for path in image_paths[:-1]]
        mask_data = np.transpose(nib.load(image_paths[-1]).get_fdata(), (2, 0, 1))
        
        # Stack modalities and convert to tensors
        vimage = np.stack(image_data, axis=0)
        vimage = torch.tensor(vimage, dtype=torch.float32)
        mask = torch.tensor(mask_data, dtype=torch.long)
        mask[mask == 4] = 3  # Adjust label values as per dataset characteristics
        
        if self.transform:
            vimage, mask = self.transform(vimage, mask)
            
        return vimage, mask
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a patient's MRI data and mask by index.
        
        :param index: Index of the patient.
        :return: Tuple of (multi-modal image tensor, mask tensor).
        """
        patient_dir = self.patients_dirs[index]
        patient_vimage, patient_mask = self.load_image(patient_dir)
        return patient_vimage, patient_mask


if __name__ == "__main__":
    data_root = "/mnt/g/DATASETS/BraTS21_original_kaggle"
    data_dir = os.path.join(data_root, "BraTS2021_00621")

    start_time = time.time()
    train_dataset = BraTS21_3D("/mnt/d/AI_Research/WS-HUB/Linux-VoxelMedix/VoxelMedix/data/raw/brats21_original/train.csv")
    print(f"Total time to initialize dataset: {time.time() - start_time:.4f} seconds")

    start_time = time.time()
    sample = train_dataset[0]
    print(f"Time to get first sample: {time.time() - start_time:.4f} seconds")
    print(sample[0].shape, sample[1].shape)