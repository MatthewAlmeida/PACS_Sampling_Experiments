from argparse import Namespace
from typing import Dict, List, Union

import h5py
import numpy as np
import os
from pathlib import Path

from dotenv import (
    load_dotenv, find_dotenv
)

def _get_sds_str(arg_ns: Namespace) -> str:
    if arg_ns.use_sds:
        return "sds"
    else:
        return "nosds"

def results_save_filename(arg_ns: Namespace) -> str:
    sds_str = _get_sds_str(arg_ns)

    return Path(
        f"results/{sds_str}/cm-random-seed-{arg_ns.random_seed}-{sds_str}.pt"
    )

def checkpoint_save_filename(arg_ns:Namespace) -> str:
    sds_str = _get_sds_str(arg_ns)

    return Path(
        f"{arg_ns.experiment_name}-{arg_ns.random_seed}-{sds_str}-" + "{epoch}"
    )

def resolve_PACS_root(pacs_root: str = None) -> Path:
    """
    If a pacs_root argument is not passed to a function,
    make sure it exists as an environment variable.
    """
    if not pacs_root:
        load_dotenv(find_dotenv())
        
        try:
            pacs_root = Path(os.environ['PACS_HOME'])
        except:
            raise KeyError(
                f"Unable to find PACS dataset root directory. Make sure "
                f"to pass the root location as an argument if the "
                f"PACS_HOME environment variable is not set in the .env file."
            )
    
    return Path(pacs_root)

def get_PACS_filenames() -> str:
    domains = ["art_painting", "cartoon", "photo", "sketch"]
    splits = ["train", "val", "test"]
    ext = ".hdf5"

    filenames = {}

    for d in domains:
        for spl in splits:
            filenames[d] = {spl: f"{d}_{spl}{ext}"}

    return filenames

def set_swmr_true_all_hdf5_files():
    """
    We use this function to flip the smwr flag on all hdf5 files.
    """

    meta = PACSMetadata()

    for filename in meta:
        with h5py.File(filename, "a", libver="latest") as domain_data:
            domain_data.swmr_mode=True


class PACSDomain():
    def __init__(self, domain_name:str, 
        splits:List=["train", "val", "test"],
        file_ext:str="hdf5"
    ) -> None:
        self._domain_name = domain_name
        self._splits = splits
        self._file_ext = "hdf5"

        self._filenames = {spl:f"{self._domain_name}_{spl}.{self._file_ext}" for spl in self._splits}

        self._generator = (fn for fn in self._filenames.values())
    
    def get_split_filename(self, split_name):
        return self._filenames[split_name]

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._generator)
        
class PACSMetadata():
    def __init__(self, pacs_root:Union[str, Path]=None) -> None:
        self.pacs_root = resolve_PACS_root(pacs_root)
        self._domain_names = ["art_painting", "cartoon", "photo", "sketch"]
        self._splits = ["train", "val", "test"]
        self._ext = ".hdf5"

        self._domains = {dname: PACSDomain(dname) for dname in self._domain_names}

        self._generator = (self.pacs_root/file_name for domain in self._domains.values() for file_name in domain)

    @property
    def domain_names(self) -> str:
        return self._domain_names

    def get_filename(self, domain_name, split_name) -> Path:
        return Path(self.pacs_root/self._domains[domain_name].get_split_filename(split_name))

    def get_filenames_except(self, held_out_domain:Union[str,int], split_name) -> List:
        try:
            if isinstance(held_out_domain, str):
                return [self.pacs_root/domain.get_split_filename(split_name) for dname, domain in self._domains.items() if not dname == held_out_domain]
            elif isinstance(held_out_domain, int):
                return [self.pacs_root/self._domains[self._domain_names[domain_idx]].get_split_filename(split_name) for domain_idx in range(len(self._domain_names)) if not domain_idx == held_out_domain]
            else:
                raise ValueError(f"Domain index must be str or int, got {held_out_domain} "
                    f"of type {type(held_out_domain)}."
                )
        except KeyError:
            raise KeyError(f"{split_name} not a valid split type; must be in 'train', 'val', 'test'")

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._generator)