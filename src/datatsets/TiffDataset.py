import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class NpzDataset(Dataset):
    def __init__(self, data_list_path):
        index = Path(data_list_path)
        with index.open(encoding="utf-8") as f:
            self.samples = [index.parent / name for name in json.load(f)]
        if not self.samples:
            raise ValueError(f"Empty dataset index: {index}")
        with np.load(self.samples[0]) as sample:
            self.num_metadata = sample["norm_metadata"].shape[0]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        with np.load(self.samples[idx]) as data:
            return {key: torch.from_numpy(data[key]) for key in
                    ("norm_metadata", "prompt_embeds", "latent_image", "cond")}


def custom_collate_fn(batch):
    return {
        "norm_metadata": torch.stack([item["norm_metadata"] for item in batch]),
        "prompt_embeds": [item["prompt_embeds"] for item in batch],
        "latent_image": torch.stack([item["latent_image"] for item in batch]),
        "cond": [torch.stack([item["cond"] for item in batch])],
    }
