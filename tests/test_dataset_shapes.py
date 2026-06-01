from __future__ import annotations

import numpy as np

from ball_rssm.data.sequence_dataset import SequenceDataset


def test_sequence_dataset_window_shapes(tmp_path) -> None:
    obs = np.random.randn(4, 11, 6).astype(np.float32)
    action = np.random.randn(4, 10, 2).astype(np.float32)
    reward = np.random.randn(4, 10, 1).astype(np.float32)
    done = np.zeros((4, 10, 1), dtype=np.float32)
    path = tmp_path / "dummy.npz"
    np.savez(path, obs=obs, action=action, reward=reward, done=done)

    dataset = SequenceDataset(path, seq_len=5, split="all")
    sample = dataset[0]

    assert sample["obs"].shape == (6, 6)
    assert sample["action"].shape == (5, 2)
    assert sample["reward"].shape == (5, 1)
    assert sample["done"].shape == (5, 1)
    assert sample["obs"].dtype.is_floating_point
    assert sample["action"].dtype.is_floating_point
