import zarr, torch
from torch.utils.data import Dataset
import numpy as np
from src.embedding_utils import generate_embeddings

class ZarrDataset(Dataset):
    """
    PyTorch Dataset for loading images from a Zarr file lazily.
    Inputs:
    - `zarr_path`: str, path to the Zarr file
    - `nepisodes`: int or None, number of episodes to load. If None, load all.
    Outputs:
    - PyTorch Dataset object
    - Single Sample shape: (C, H, W)
    """
    def __init__(self, zarr_path, nepisodes = None):
        self.store = zarr.open(zarr_path, mode="r")
        self.episode_ends = self.store["meta/episode_ends"][:] if nepisodes is None else self.store["meta/episode_ends"][:nepisodes]
        self.episode_starts = np.concatenate(([0], self.episode_ends[:-1]))
        self.z = self.store["data/img"][:self.episode_ends[-1]]
        self.length = self.z.shape[0]
        self.nepisodes = len(self.episode_ends)

    def __len__(self):
        return self.length
    
    def _preprocess(self, x):
        if len(x.shape)==3:
            x = np.ascontiguousarray(np.transpose(x, (2, 0, 1)))/255.0  # Normalize to [0, 1]
        elif len(x.shape)==4:
            x = np.ascontiguousarray(np.transpose(x, (0, 3, 1, 2)))/255.0  # Normalize to [0, 1]
        return torch.from_numpy(x)

    def __getitem__(self, idx):
        x = self.z[idx] 
        x = self._preprocess(x)
        return x
    

class ZarrPairwiseDataset(ZarrDataset):
    """
    PyTorch Dataset for loading sequences of images from a Zarr file lazily.
    Inputs:
    - `zarr_path`: str, path to the Zarr file
    - `nepisodes`: int or None, number of episodes to load. If None, load all.
    Outputs:
    - PyTorch Dataset object
    - Single Sample shape: (2, C, H, W)
    """
    def __init__(self, zarr_path, nepisodes=None):
        super().__init__(zarr_path, nepisodes)
        self.valid_anchors = np.concatenate([np.arange(start, end - 1) for start, end in zip(self.episode_starts, self.episode_ends)])
        self.length = len(self.valid_anchors)

    def __getitem__(self, idx):
        xi = super().__getitem__(idx)
        xi1 = super().__getitem__(idx + 1)
        return torch.stack([xi, xi1], dim=0)
    

class ZarrTripletDataset(ZarrPairwiseDataset):
    """
    PyTorch Dataset for loading triplets of images from a Zarr file lazily.
    Inputs:
    - `zarr_path`: str, path to the Zarr file
    - `nepisodes`: int or None, number of episodes to load. If None, load all.
    - `min_distance`: int, minimum temporal distance between anchor and negative samples
    Outputs:
    - PyTorch Dataset object
    - Single Sample shape: (3, C, H, W)
    """
    def __init__(self, zarr_path, nepisodes=None, min_distance=10):
        super().__init__(zarr_path, nepisodes)
        self.min_distance = min_distance

    def __getitem__(self, idx):
        vid_idx = np.searchsorted(self.episode_ends, idx, side='right')
        idx_j = np.random.randint(self.episode_starts[vid_idx], self.episode_ends[vid_idx])
        while abs(idx - idx_j) < self.min_distance:
            idx_j = np.random.randint(self.episode_starts[vid_idx], self.episode_ends[vid_idx])
        xi, xj = super().__getitem__(idx), self._preprocess(self.z[idx_j])
        return torch.stack([xi[0], xi[1], xj], dim=0)


if __name__ == "__main__":

    dataset = ZarrDataset("pusht/pusht_cchi_v7_replay.zarr", nepisodes=5)
    print("Dataset length:", len(dataset))
    sample = dataset[0]
    print("Sample shape:", sample.shape)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in dataloader:
        print("Batch shape:", batch.shape)
        break

    pairwise_dataset = ZarrPairwiseDataset("pusht/pusht_cchi_v7_replay.zarr", nepisodes=5)
    pair_sample = pairwise_dataset[0]
    print("Pair Sample shapes:", pair_sample.shape)

    triplet_dataset = ZarrTripletDataset("pusht/pusht_cchi_v7_replay.zarr", nepisodes=5, min_distance=10)
    triplet_sample = triplet_dataset[0]
    print("Triplet Sample shapes:", triplet_sample.shape)

    dataloader = torch.utils.data.DataLoader(triplet_dataset, batch_size=4, shuffle=True)
    for batch in dataloader:
        print("Batch shape:", batch.shape)
        break