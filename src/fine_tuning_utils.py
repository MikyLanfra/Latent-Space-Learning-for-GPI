import torch
import numpy as np
import torch.nn as nn
from torch import optim

def get_pos_neg_samples(dataset, add_steps=0, mode='initial'):
    """
    Given a ZarrDataset, randomly select one episode to extract all its frames as positive samples and frames from other episodes as negative samples.
    Inputs:
    - `dataset`: ZarrDataset object
    - `add_steps`: int, number of additional frames to consider after each episode start for negative sampling
    Outputs:
    - `pos_indices`: np.ndarray of shape (P,), indices of positive samples
    - `neg_indices`: np.ndarray of shape (N,), indices of negative samples
    """
    i = np.random.randint(0, dataset.nepisodes)
    start, end = dataset.episode_starts[i], dataset.episode_ends[i]
    pos_indices = np.arange(start, end)

    if mode == 'initial':
        other_starts = np.delete(dataset.episode_starts, i)
        neg_indices = (other_starts[:, None] + np.arange(add_steps + 1)).ravel()
        neg_indices = neg_indices[neg_indices < len(dataset.z)]

    else:
        checkpoints = dataset.episode_starts
        checkpoints = np.append(checkpoints, len(dataset.z))  # Add end index for the last episode
        num_episodes = len(checkpoints)
        num_neg = num_episodes * add_steps  # Total number of negative samples to draw
        min_ep_len = np.min(np.diff(checkpoints))  # Minimum episode length to avoid index errors
        neg_indices = []
        for i in range(num_neg):
            while True:
                ep_idx = np.random.randint(0, num_episodes-1)  # Random episode index
                if ep_idx != i:  # Ensure it's not the same episode
                    curr_len = checkpoints[ep_idx + 1] - checkpoints[ep_idx]
                    j = np.random.randint(min_ep_len//4, 3*min_ep_len//4)
                    neg_indices.append(checkpoints[ep_idx] + j)
                    break

    return pos_indices, neg_indices



def compute_similarities(p_norm, n_norm):
    """
    Given normalized positive and negative samples, compute similarity matrices.

    Inputs:
    - `p_norm`: torch.Tensor of shape (M, D), normalized positive samples
    - `n_norm`: torch.Tensor of shape (K, D), normalized negative samples

    Outputs:
    - `sim_pp`: torch.Tensor of shape (M, M), similarity matrix among positives with diagonal set to -inf
    - `sim_pn`: torch.Tensor of shape (M, K), similarity matrix between positives and negatives
    """
    sim_pp = torch.matmul(p_norm, p_norm.T) # Similarity among positives
    sim_pp.fill_diagonal_(-float('inf')) # Exclude self-similarity
    sim_pn = torch.matmul(p_norm, n_norm.T) # Similarity between positives and negatives
    return sim_pp, sim_pn



def temporal_weights(num_pos, lambda_, max_distance=None, device='cpu'):
    """
    Compute temporal weights for positive samples based on their indices.
    Weights decay exponentially with distance.

    Inputs:
    - `num_pos`: int, number of positive samples
    - `lambda_`: float, decay parameter for the exponential function
    - `max_distance`: int or None, maximum distance to consider; if None, set to num_pos
    - `device`: str or torch.device, device to create the tensor on

    Outputs:
    - `weights`: torch.Tensor of shape (num_pos, num_pos), temporal weight matrix with zeros on the diagonal
    """

    if max_distance is None: max_distance = num_pos # Default to num_pos if not specified

    idx = torch.arange(num_pos, dtype=torch.float32, device=device)  # Index range, (num_pos,)

    distances = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))  # Pairwise absolute differences, (num_pos, num_pos)

    distances[distances > max_distance] = float('inf')  # Cap distances at max_distance
    weights = torch.exp(-distances / lambda_) # Exponential decay
    weights.fill_diagonal_(0)  # no self-weight

    return weights



def compute_loss_epoch(model, dataset, lambda_, temp, device, max_distance=None, add_steps=0, uniform=False, mode='initial'):
        """
        Computes the loss for one epoch using contrastive loss with temporal weighting.
    
        Input:
        - `data`: Input tensor of shape (num_samples, input_dim).
        - `checkpoints`: List or tensor of checkpoint indices.
        - `lambda_`: Decay parameter for temporal weights.
        - `max_distance`: Maximum temporal distance for weighting (optional).
        - `add_steps`: Additional steps to consider for positive samples.
        - `uniform`: Whether to use uniform sampling for loss computation. If all episodes converge to the same point, set to True.

        Output:
        - Computed loss for the epoch.
        """
        if model.embeddings is None: model.get_embeddings(dataset, device)

        pos_idx, neg_idx = get_pos_neg_samples(dataset, add_steps=add_steps, mode=mode) # Get positive and negative samples
        pos = model.embeddings[pos_idx] # Positive samples
        neg = model.embeddings[neg_idx] # Negative samples
        pos = nn.functional.normalize(pos, dim=1) # Normalize positive samples
        neg = nn.functional.normalize(neg, dim=1) # Normalize

        pos = torch.tensor(pos, requires_grad=True).to(device)
        neg = torch.tensor(neg, requires_grad=True).to(device)
        pos = model.head_model(pos) # Pass through head model
        neg = model.head_model(neg) # Pass through head model

        if uniform: max_idx = np.random.randint(0, pos.shape[0]) # Randomly select a max_idx for uniform sampling
        else: max_idx = pos.shape[0]
            
        sim_pp, sim_pn = compute_similarities(pos, neg) # Compute similarity matrices
        sim_matrix = torch.cat((sim_pp, sim_pn), dim=1) # Concatenate positive and negative similarity matrices

        # Softmax over the similarity matrix
        exp_matrix = torch.exp(sim_matrix/temp) # Exponentiate similarity matrix
        sum_exp = torch.sum(exp_matrix, dim=1, keepdim=True) # Sum over rows for normalization
        prob_matrix = (exp_matrix / sum_exp)[:max_idx, :pos.shape[0]] # Probability matrix for selected subset
        log_prob = torch.log(prob_matrix + 1e-8) # Log probabilities with numerical stability

        weights = temporal_weights(pos.shape[0], lambda_, max_distance=max_distance, device=device)[:max_idx,:] # Temporal weights
        loss = -torch.sum((weights * log_prob)) / max_idx # Weighted loss computation

        return loss


def train_loop(model, dataset, lambda_, temp, device, max_distance=None, add_steps=0, uniform=False, epochs=10, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = compute_loss_epoch(model, dataset, lambda_, temp, device, max_distance, add_steps, uniform, mode='random')
        loss.backward()
        optimizer.step()
        n = max(1, epochs // 10)  # Print progress every 10% of epochs
        if epoch % n == 0 or epoch == epochs - 1: print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")



if __name__ == "__main__":
    from dataset import ZarrDataset
    dataset = ZarrDataset("pusht/pusht_cchi_v7_replay.zarr", nepisodes=10)
    positives, negatives = get_pos_neg_samples(dataset, add_steps=5)
    print("Positives shape:", positives.shape)
    print("Negatives shape:", negatives.shape)

    pos_fake = torch.randn((len(positives), 128))
    neg_fake = torch.randn((len(negatives), 128))
    p_norm = pos_fake / torch.norm(pos_fake, dim=1, keepdim=True)
    n_norm = neg_fake / torch.norm(neg_fake, dim=1, keepdim=True)
    sim_pp, sim_pn = compute_similarities(p_norm, n_norm)
    print("Sim PP shape:", sim_pp.shape)
    print("Sim PN shape:", sim_pn.shape)

