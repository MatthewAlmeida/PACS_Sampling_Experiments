from typing import List, Union

import h5py
import numpy as np
import os

from pathlib import Path
from .pacs_utils import (
    PACSMetadata
)

import torch

def _get_torchvision_means_stds():
    """
    Pytorch's torchvision pretrained models use the following
    normalization statistics by channel. Unfortunately, we just
    have to hard-code them in somewhere.
    """
    return {
        "means": np.array([0.485, 0.456, 0.406]), 
        "stds": np.array([0.229, 0.224, 0.225])
    }

def _preprocess_image(img:np.ndarray, normalize:bool, means:np.ndarray, stds:np.ndarray) -> np.ndarray:
    """
    Perform the necessary preprocessing to meet the expectations of the 
    pytorch torchvision library.
    """
    # Channel order BGR -> RGB
    wip_img = img[:,:,[2,1,0]].astype(float)

    if normalize:
        # features -> [0,1]
        wip_img /= 255.0

        #Crop to 224 x 224
        wip_img_crop = wip_img[1:-2, 1:-2, :]

        # Normalize to means and standard deviations based on pytorch 
        # pretrained model documentation

        wip_out = (wip_img_crop - means) / stds
    else:
        wip_out = wip_img

    # Reorganize HWC axis order to CHW
    return np.transpose(wip_out, (2,0,1))

def _preprocess_label(lbl):
    """
    The provided datasets have labels that are 1-indexed. We convert them
    to 0-index here.
    """
    return lbl - 1

class PACSDatasetSingleDomain(torch.utils.data.Dataset):
    def __init__(self, 
        domain_name:str, split_name:str, 
        normalize: bool = True,
        pacs_root: Union[str, Path] = None,
    ) -> None:
        super().__init__()
        self._domain_name = domain_name
        self._split_name = split_name
        self._normalize = normalize

        self._meta = PACSMetadata(pacs_root=pacs_root)

        # Get the location of the hdf5 file containing the data for this domain
        self._filename = self._meta.get_filename(self._domain_name, self._split_name)

        # From pytorch, for pretrained model
        
        mean_std_dict = _get_torchvision_means_stds()

        self._means = mean_std_dict["means"]
        self._stds = mean_std_dict["stds"]

        # How many examples are in this domain?
        with h5py.File(self._filename, "r") as domain_data:
            self._n_examples = domain_data["images"].shape[0]

    def __len__(self):
        return self._n_examples

    def __getitem__(self, example_idx):
        with h5py.File(self._filename, "r") as domain_data:
            return (
                _preprocess_image(
                    domain_data["images"][example_idx],
                    normalize=self._normalize,
                    means=self._means,
                    stds=self._stds
                ),
                _preprocess_label(
                    domain_data["labels"][example_idx]
                )                
            )

class PACSDatasetMultipleDomain(torch.utils.data.ConcatDataset):
    def __init__(self, holdout_domain:str, split_name:str, 
        normalize:bool=True, pacs_root: Union[str, Path] = None
    ) -> None:
        self._holdout_domain = holdout_domain
        self._split_name = split_name
        self._normalize = normalize
        self._meta = PACSMetadata(pacs_root=pacs_root)

        self._domain_names = [dname for dname in self._meta.domain_names if not dname == holdout_domain]

        # Assemble domains as a list of Dataset objects
        datasets = []
        self._dataset_lengths = []

        for dname in self._domain_names:
            domain = PACSDatasetSingleDomain(
                    dname, split_name, normalize, pacs_root
                )

            datasets.append(domain)
            self._dataset_lengths.append(len(domain))

        si_cursor = 0
        self._split_indices = []

        for dom_len in self._dataset_lengths:
            self._split_indices.append(dom_len + si_cursor)
            si_cursor += dom_len
        
        # Inherits constructor from torch's ConcatDataset. Passing the 
        # list of datasets allows us to use the superclass definitions
        # of __len__ and __getitem__. This class exists to hold on to 
        # metadata about the original domains in the ConcatDataset.
        super().__init__(
            datasets
        )

    @property
    def split_indices(self):
        return self._split_indices

    @property
    def domain_names(self):
        return self._domain_names

    @property
    def split_name(self):
        return self._split_name
