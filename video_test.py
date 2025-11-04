import matplotlib.pyplot as plt
import matplotlib.animation as animation
import zarr, torch

from sklearn.manifold import TSNE
import numpy as np


def show_as_video(images, pos, act, interval=200):
    images = images / 255
    pos = pos * 96 / 512
    act = act * 96 / 512

    fig, ax = plt.subplots()
    im = ax.imshow(images[0].squeeze(), animated=True)
    arrow = ax.arrow(pos[0, 0], pos[0, 1],
                     act[0, 0] - pos[0, 0],
                     act[0, 1] - pos[0, 1],
                     head_width=2, head_length=2,
                     fc='r', ec='r')

    def update(frame):
        im.set_array(images[frame].squeeze())
        nonlocal arrow
        # remove old arrow
        arrow.remove()
        # draw new arrow
        arrow = ax.arrow(pos[frame, 0], pos[frame, 1],
                         act[frame, 0] - pos[frame, 0],
                         act[frame, 1] - pos[frame, 1],
                         head_width=2, head_length=2,
                         fc='r', ec='r')
        return [im, arrow]

    ani = animation.FuncAnimation(
        fig, update, frames=len(images), interval=interval, blit=True, repeat=True
    )
    plt.show()


def get_episode_interval(endpoints, i):
    if i < len(endpoints) - 1:
        return endpoints[i], endpoints[i + 1]
    return None


def show_video_with_tsne(images, pos, act, embeddings, interval=200):
    """
    images: list/array of shape (T, H, W) or (T, H, W, C)
    pos:    array (T, 2) of positions
    act:    array (T, 2) of actions
    embeddings: array (T, D) to project with t-SNE
    """

    # Normalize image input
    images = images / 255.0
    pos = pos * 96 / 512
    act = act * 96 / 512

    # t-SNE projection
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    print("t-SNE completed.")

    # Create figure with 2 panels
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))

    # --- Left panel: video ---
    im = ax0.imshow(images[0].squeeze(), animated=True)
    arrow = ax0.arrow(pos[0, 0], pos[0, 1], act[0, 0] - pos[0, 0], act[0, 1] - pos[0, 1], head_width=2, head_length=2, fc='r', ec='r')
    ax0.set_title("Video + Action")

    # --- Right panel: TSNE trajectory ---
    ax1.plot(embeddings_2d[:, 0], embeddings_2d[:, 1], 'o-', markersize=2, alpha=0.5)
    current_point, = ax1.plot(embeddings_2d[0, 0], embeddings_2d[0, 1], 'ro', markersize=8, label="Current frame")
    ax1.legend()
    ax1.set_title("t-SNE Embedding Trajectory")

    def update(frame):
        nonlocal arrow
        # Update video image
        im.set_array(images[frame].squeeze())

        # Update arrow
        arrow.remove()
        arrow = ax0.arrow(pos[frame, 0], pos[frame, 1],
                          act[frame, 0] - pos[frame, 0],
                          act[frame, 1] - pos[frame, 1],
                          head_width=2, head_length=2,
                          fc='r', ec='r')

        # Update highlight on embedding
        ax1.cla()
        ax1.plot(embeddings_2d[:, 0], embeddings_2d[:, 1], 'o-', markersize=2, alpha=0.5)
        current_point, = ax1.plot(embeddings_2d[frame, 0], embeddings_2d[frame, 1], 'ro', markersize=8, label="Current frame")
        ax1.set_title("t-SNE Embedding Trajectory")
        ax1.legend()

        return [im, arrow, current_point]

    ani = animation.FuncAnimation(
        fig, update, frames=len(images), interval=interval, blit=False, repeat=True)

    plt.show()
    return ani


if __name__ == "__main__":
    store = zarr.open("pusht/pusht_cchi_v7_replay.zarr", mode="r")
    endpoints = store["meta/episode_ends"]
    endpoints = np.insert(endpoints, 0, 0)

    i = 2
    emb_file = "embeddings_tuned_temp_tiny.pth"

    start, end = get_episode_interval(endpoints, i)
    images = store["data/img"][start:end]
    positions = store["data/state"][start:end, :2]
    actions = store["data/action"][start:end]
    embeddings = torch.load(emb_file).detach()[start:end]

    # show_as_video(images, positions, actions, interval=50)  # 100 ms per frame
    show_video_with_tsne(images, positions, actions, embeddings, interval=50)  # 100 ms per frame