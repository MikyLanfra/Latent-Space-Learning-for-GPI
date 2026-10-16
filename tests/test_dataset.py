import numpy as np
import pytest
import zarr

from src.dataset import ZarrPairwiseDataset, ZarrTripletDataset

EPISODE_ENDS = np.array([8, 18, 30])  # every anchor needs a frame >= min_distance away
H = W = 4


@pytest.fixture
def zarr_path(tmp_path):
    # pixel value of frame k is k, so every sample reveals the frame indices it came from
    n = EPISODE_ENDS[-1]
    imgs = np.broadcast_to(
        np.arange(n, dtype=np.float32)[:, None, None, None], (n, H, W, 3)
    )
    root = zarr.open_group(str(tmp_path / "toy.zarr"), mode="w")
    root.create_array("data/img", data=np.ascontiguousarray(imgs))
    root.create_array("meta/episode_ends", data=EPISODE_ENDS)
    return str(tmp_path / "toy.zarr")


def frame_of(x):
    return int(round(float(x[0, 0, 0]) * 255.0))


def episode_of(frame):
    return int(np.searchsorted(EPISODE_ENDS, frame, side="right"))


def test_pairs_are_consecutive_and_within_episode(zarr_path):
    ds = ZarrPairwiseDataset(zarr_path)
    assert len(ds) == EPISODE_ENDS[-1] - len(EPISODE_ENDS)
    seen = set()
    for i in range(len(ds)):
        pair = ds[i]
        assert pair.shape == (2, 3, H, W)
        a, b = frame_of(pair[0]), frame_of(pair[1])
        assert b == a + 1 and episode_of(a) == episode_of(b)
        seen.add(a)
    assert seen == set(ds.valid_anchors.tolist())  # every valid anchor is reachable


def test_triplet_negative_from_same_episode(zarr_path):
    ds = ZarrTripletDataset(zarr_path, min_distance=3)
    np.random.seed(0)
    for i in range(len(ds)):
        trip = ds[i]
        assert trip.shape == (3, 3, H, W)
        a, b, n = (frame_of(trip[k]) for k in range(3))
        assert b == a + 1
        assert episode_of(n) == episode_of(a) and abs(n - a) >= 3
