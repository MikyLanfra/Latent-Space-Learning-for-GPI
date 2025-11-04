import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from matplotlib.collections import LineCollection



def emb_hmap(embeddings, title=None):
    """
    Given a set of embeddings, compute and plot the heatmap of their pairwise similarities.
    Inputs:
    - `embeddings`: torch.Tensor of shape (N, D), where N is the number of embeddings and D is their dimensionality
    Outputs:
    - Heatmap plot of pairwise similarities
    - Prints the minimum similarity value
    """
    embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
    emb_h = torch.matmul(embeddings, embeddings.T)
    sns.heatmap(np.abs(emb_h.numpy()), cmap='viridis', vmin=0, vmax=1)
    print(f"Min_similarity: {np.abs(emb_h).min().item():.4f}")
    if title: plt.title(title)
    plt.show()



def plot_2d_embeddings(embeddings, title):
    """
    Given a set of embeddings, reduce them to 2D using t-SNE and plot them with line segments colored by their lengths.
    Inputs:
    - `embeddings`: np.ndarray of shape (N, D), where N is the number of embeddings and D is their dimensionality
    - `title`: str, title for the plot
    Outputs:
    - 2D plot of embeddings with line segments colored by lengths
    - Histogram of segment lengths
    """
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Prepare line segments between consecutive points
    points = embeddings_2d.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Compute segment lengths
    lengths = np.linalg.norm(segments[:, 1, :] - segments[:, 0, :], axis=1)

    # Normalize lengths for colormap
    norm = plt.Normalize(lengths.min(), lengths.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(lengths)
    lc.set_linewidth(2)

    # Create a figure with two panels: embedding + histogram
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])

    # Left: embedding plot
    ax0 = fig.add_subplot(gs[0])
    ax0.add_collection(lc)
    ax0.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='k', s=10)
    ax0.autoscale()
    ax0.set_title(title)

    # Colorbar for line lengths
    cbar = plt.colorbar(lc, ax=ax0, label="Segment length")

    # Right: histogram of segment lengths
    ax1 = fig.add_subplot(gs[1])
    ax1.hist(lengths, bins=20, color="steelblue", edgecolor="black")
    ax1.set_title("Segment Length Distribution")
    ax1.set_xlabel("Length")
    ax1.set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()



def visualize_similarities(anchor, data, img_anchor=None, imgs=None):
    """
    Given an anchor vector and a set of data vectors, compute and plot the cosine similarities.
    If images are provided, display the anchor image and the most similar image.
    Inputs:
    - `anchor`: torch.Tensor of shape (D,), the anchor vector
    - `data`: torch.Tensor of shape (N, D), the set of data vectors
    - `img_anchor`: torch.Tensor of shape (C, H, W), the anchor image (optional)
    - `imgs`: list of torch.Tensor of shape (C, H, W), the set of images corresponding to data vectors (optional)
    Outputs:
    - Bar plot of cosine similarities
    - Displays anchor image and most similar image if provided
    - Returns the index of the most similar data vector
    """
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    similarities = cos(anchor.unsqueeze(0), data)

    #Barplot of similarities    
    plt.figure(figsize=(10,5))
    plt.bar(np.arange(len(similarities)), similarities.cpu().numpy())
    plt.xlabel("Index")
    plt.ylabel("Cosine Similarity")
    plt.title("Cosine Similarities to Anchor")
    plt.show()

    max_idx = torch.argmax(similarities).item()
    print(f"Most similar index: {max_idx}, Similarity: {similarities[max_idx].item():.4f}")
    if imgs is not None:
        plt.subplots(1,2, figsize=(8,4))
        plt.subplot(1,2,1)
        plt.imshow(img_anchor.permute(1,2,0)/255)
        plt.title("Anchor Image")
        plt.subplot(1,2,2)
        plt.imshow(imgs[max_idx]/255)
        plt.title(f"Most Similar Image (Index: {max_idx})")
        plt.tight_layout()
        plt.show()
    return max_idx



def plot_rec(orig, recons):
    """
    Given original and reconstructed images, plot them side by side for comparison.
    Inputs:
    - `orig`: torch.Tensor of shape (C, H, W), the original image
    - `recons`: torch.Tensor of shape (C, H, W), the reconstructed image
    Outputs:
    - Side-by-side plot of original and reconstructed images
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(orig.squeeze().cpu(), cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Image")
    plt.imshow(recons.squeeze().cpu(), cmap='gray')
    plt.axis('off')
    plt.show()