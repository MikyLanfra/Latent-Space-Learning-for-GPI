import torch
import numpy as np
import matplotlib.pyplot as plt

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