import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

def smoothness_metrics(embs, plot=True):
    data = torch.sqrt(torch.sum((embs[1:] - embs[:-1])**2, dim=1)).cpu().numpy()
    avg_l2 = np.mean(data)
    if plot:
        plt.figure()
        plt.plot(data/avg_l2)
        plt.title("L2 Distance Between Consecutive Embeddings")
        plt.xlabel("Frame Index")
        plt.ylabel("L2 Distance")
        plt.show()
        print("Variance of L2 distances:", np.var(data))
    return data/avg_l2, avg_l2, np.var(data)


def smoothness_ratio(embs):
    """
    Scale-invariant smoothness of one embedded trajectory: mean consecutive step divided by
    mean pairwise distance. Lower is smoother; uniformly rescaling `embs` leaves it unchanged.
    Inputs:
    - `embs`: np.ndarray or torch.Tensor of shape (T, D), embeddings of one episode in time order
    Outputs:
    - float
    """
    embs = np.asarray(embs, dtype=np.float64)
    steps = np.linalg.norm(np.diff(embs, axis=0), axis=1)
    iu = np.triu_indices(len(embs), k=1)
    return steps.mean() / cdist(embs, embs)[iu].mean()


def time_order_correlation(embs):
    """
    Spearman correlation between latent distance and temporal distance |i - j| over all frame
    pairs of one episode. Higher means latent distance tracks elapsed time; scale-invariant.
    Inputs:
    - `embs`: np.ndarray or torch.Tensor of shape (T, D), embeddings of one episode in time order
    Outputs:
    - float in [-1, 1]
    """
    embs = np.asarray(embs, dtype=np.float64)
    iu = np.triu_indices(len(embs), k=1)
    return spearmanr(iu[1] - iu[0], cdist(embs, embs)[iu]).statistic
