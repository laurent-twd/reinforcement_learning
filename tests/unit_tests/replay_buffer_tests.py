from src.data.replay_buffer import ReplayBuffer
import torch


def test_replay_buffer():

    replay_buffer = ReplayBuffer(100)
    sample = torch.FloatTensor([0.0, 1.0])
    _ = [
        replay_buffer.add_experience(sample, sample, sample, sample, sample)
        for _ in range(100)
    ]
    assert replay_buffer.buffer_size() == 100
    new_sample = torch.FloatTensor([0.0, 1.0])
    replay_buffer.add_experience(
        new_sample, new_sample, new_sample, new_sample, new_sample
    )
    assert (replay_buffer.buffer["state"][-1] == new_sample).all()
    assert (replay_buffer.buffer["state"][0] == sample).all()
