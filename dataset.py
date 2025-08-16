import argparse
import math
import os
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from dask.dataframe.multi import required
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class PathDataset(Dataset):
    def __init__(self,
        use_synth: bool = False,
        synth_n: int = 2000,
        synth_len_min: int = 8,
        synth_len_max: int = 64,
        vocab_hash_size: int = 32768,
        num_feat_dim: int = 4,
        data_path: Optional[str] = None,
        predict_delay: bool = True,
        predict_power: bool = False
                 ):
        self.vocab_hash_size = vocab_hash_size
        self.num_feat_dim = num_feat_dim
        self.predict_delay = predict_delay
        self.predict_power = predict_power

        if use_synth:
            self.samples = self._gen_synth(
                synth_n, synth_len_min, synth_len_max, num_feat_dim
            )


    def _gen_synth(self, n: int, Lmin: int, Lmax: int, dnum: int):
        samples = []
        for _ in range(n):
            L = random.randint(Lmin, Lmax)
            pip_seq = [random.randint(0, self.vocab_hash_size - 1) for _ in range(L)]
            tile_seq = [random.randint(0, self.vocab_hash_size - 1) for _ in range(L)]

    def _load_real(self, path:str):
        assert pd is not None, "need pands read csv/parquet"
        if path.endswith('.parquet'):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        required = ['pip_seq', 'tile_seq', 'delay_ns']
        for k in required:

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx:int):
        s = self.samples[idx]
        return s