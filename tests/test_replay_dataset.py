import json
from pathlib import Path

import numpy as np

from mini_groot_sonic.data.episode_writer import EpisodeWriter
from mini_groot_sonic.data.replay_dataset import ReplayWindowDataset


def _episode_arrays(offset: float = 0.0):
    t = 3
    identity = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], np.float32), (t, 1))
    slots = np.zeros((t, 6, 7), np.float32)
    slots[..., 3] = 1.0
    return {
        "q": np.full((t, 29), offset, np.float32),
        "qdot": np.zeros((t, 29), np.float32),
        "root_pos": np.zeros((t, 3), np.float32),
        "root_quat": identity,
        "root_linvel": np.zeros((t, 3), np.float32),
        "root_angvel": np.zeros((t, 3), np.float32),
        "token": np.arange(t * 64, dtype=np.float32).reshape(t, 64) + offset,
        "goal_slots": slots,
    }


def test_replay_state_and_token_are_causally_aligned(tmp_path: Path):
    writer = EpisodeWriter(tmp_path)
    writer.write("a", _episode_arrays(2.0), {"caption": "walk", "captions": ["walk"]})
    ds = ReplayWindowDataset(tmp_path, horizon=3, samples_per_episode=1, goal_probabilities=(1, 0, 0, 0))
    sample = ds[0]
    assert sample["state"].shape == (68,)
    assert float(sample["state"][0]) == 2.0
    assert float(sample["actions"][0, 0]) == 2.0
    slots = sample["goal"].reshape(6, 8)
    assert np.allclose(slots.numpy(), 0.0)


def test_episode_writer_replaces_metadata_instead_of_duplicating(tmp_path: Path):
    writer = EpisodeWriter(tmp_path)
    writer.write("a", _episode_arrays(), {"caption": "first"})
    writer.write("a", _episode_arrays(1.0), {"caption": "second"})
    rows = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["caption"] == "second"


def test_replay_split_keeps_actor_groups_disjoint(tmp_path: Path):
    writer = EpisodeWriter(tmp_path)
    for actor in ("actor_a", "actor_b"):
        for index in range(2):
            writer.write(
                f"{actor}_{index}",
                _episode_arrays(float(index)),
                {"caption": "walk", "actor_uid": actor},
            )
    train = ReplayWindowDataset(tmp_path, split="train", samples_per_episode=1)
    val = ReplayWindowDataset(tmp_path, split="val", samples_per_episode=1)
    train_actors = {row["actor_uid"] for row in train.rows}
    val_actors = {row["actor_uid"] for row in val.rows}
    assert train_actors
    assert val_actors
    assert train_actors.isdisjoint(val_actors)
