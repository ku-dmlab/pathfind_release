from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import numpy.typing as npt
import torch

from env.online import City

# TODO: Move append method of ReplayBuffer to ReplaySample.
# TODO: Make ReplayBuffer ABC.

@dataclass
class ReplaySample:
    """Sample from EpisodeBuffer"""
    
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    terminateds: torch.Tensor
    cities: npt.NDArray
    
class ReplayBuffer:
    """
    Replay buffer.
    
    This class pre-allocates buffer_size-sized NumPy array at the construction. It also
    assumes that the observation is flat and the action space is discrete.

    :param buffer_size: The number of episodes to store.
    :param num_actions: The number of discrete actions.
    :param observation_size: The size of the observation.
    :param device: A device of returned PyTorch tensors.
    :param seed: A seed for random sampling.
    """
    
    def __init__(
        self,
        buffer_size: int,
        observation_size: int,
        device: torch.device,
        seed: int | None = None,
    ):
        self.device = device
        self.rng = np.random.default_rng(seed)

        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False

        zeros32 = partial(np.zeros, dtype=np.float32)
        self.observations = zeros32((self.buffer_size, observation_size))
        self.actions = np.zeros((self.buffer_size,), dtype=np.int64)
        self.rewards = zeros32((self.buffer_size,))
        self.next_observations = zeros32((self.buffer_size, observation_size))
        self.terminateds = zeros32((self.buffer_size,))
        self.cities = np.empty((self.buffer_size,), dtype=City)
        
    def reset(self):
        """Reset the buffer.

        This method does not clear the pre-allocated buffers. Instead, it resets the
        information regarding indexes and overwrite as if there is nothing stored in
        the buffer.
        """
        self.pos = 0
        self.full = False

    def size(self) -> int:
        """Return the number of episodes stored."""
        if self.full:
            return self.buffer_size
        return self.pos

    def append(
        self,
        observation: npt.NDArray,
        action: int,
        reward: float,
        next_observation: npt.NDArray,
        terminated: bool,
        city: City,
    ):
        """Append a transition."""

        self.observations[self.pos] = np.copy(observation)
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_observations[self.pos] = np.copy(next_observation)
        self.terminateds[self.pos] = float(terminated)
        self.cities[self.pos] = city

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def last(self, t: int) -> ReplaySample:
        data = tuple(
            map(
                self.to_torch,
                (
                    self.observations[self.pos - t + 1 : t + 1],
                    self.actions[self.pos - t + 1 : t + 1],
                    self.rewards[self.pos - t + 1 : t + 1],
                    self.next_observations[self.pos - t + 1 : t + 1],
                    self.terminateds[self.pos - t + 1 : t + 1],
                ),
            )
        )
        return ReplaySample(*data, cities=self.cities[self.pos - t + 1 : t + 1])

    def sample(self, size: int) -> ReplaySample:
        """Sample episodes.

        The episodes are truncated to match the maximum length of episodes in the
        sample. Therefore, the returned episodes might be shorter than the time limit.

        :param size: The size of a sample.
        :returns: A dataclass for a sample.
        """

        inds = self.rng.integers(0, self.size(), (size,))
        data = tuple(
            map(
                self.to_torch,
                (
                    self.observations[inds],
                    self.actions[inds],
                    self.rewards[inds],
                    self.next_observations[inds],
                    self.terminateds[inds],
                ),
            )
        )
        return ReplaySample(*data, cities=self.cities[inds],)

    def to_torch(self, arr: np.ndarray) -> torch.Tensor:
        """Convert a numpy array to a PyTorch tensor"""
        return torch.from_numpy(arr).to(device=self.device)