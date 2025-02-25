from __future__ import annotations

import dataclasses
import os
import pickle
import random
import zipfile
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_

from gymnasium.spaces import Box, Discrete, Space

from agents.dqn.mlp import QNet
from buffers import ReplayBuffer
from env.common import City, PathfindID, Coordinate
from env.online import PathfindEnv
from env.service import TooManyRetriesError
from utils.file import makedir_with_warning
from utils.logger import DummyLogger, Logger

from tqdm import tqdm

def get_observation_size(observation_space: Space) -> int:
    if not isinstance(observation_space, Box):
        raise ValueError()
    return int(observation_space.shape[0])


def get_action_count(action_space: Space) -> int:
    if not isinstance(action_space, Discrete):
        raise ValueError()
    return int(action_space.n)

class Agent:
    """DQN with the simple MLP, assuming one-city one-text setting."""
    def __init__(
        self,
        seed: int,
        environment: PathfindEnv,
        num_hidden: int,
        hidden_dim: int,
        device: str,
        logger: Logger = DummyLogger(),
        batch_size: int = 32,
        buffer_size: int = 5000,
        gamma: float = 0.99,
        lr: float = 1e-4,
        adam_betas: tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
        weight_decay: float = 0.0,
        target_tau: float = 0.005,
    ):
        self.environment = environment
        self.per_city_time_limit = environment.per_city_time_limit
        
        self.observation_size = get_observation_size(environment.observation_space) + 1
        self.action_count = get_action_count(environment.action_space)
        
        self._set_seed(seed)
        
        self.num_hidden = num_hidden
        self.hidden_dim = hidden_dim

        self.device_type = device
        self.device = torch.device(device)
        
        self.logger = logger

        self.gamma = gamma
        self.qnet_tau = target_tau
        
        self.qnet_online = QNet(
            self.observation_size, self.action_count, num_hidden, hidden_dim
        ).to(device)
        self.qnet_target = QNet(
            self.observation_size, self.action_count, num_hidden, hidden_dim
        ).to(device)
        self.qnet_target.load_state_dict(self.qnet_online.state_dict())
        
        self.optimizer = optim.Adam(
            self.qnet_online.parameters(),
            lr=lr,
            betas=adam_betas,
            eps=adam_eps,
            weight_decay=weight_decay,
        )
        self.crit = torch.nn.MSELoss()

        self.buffer = ReplayBuffer(
            buffer_size,
            self.observation_size,
            self.device,
            seed=seed,
        )
        
        self.batch_size = batch_size

        self.state_action_visitations: dict[PathfindID, list[int]] = {}
        self.exploration_coefficient = 2.0
        self.exploration_prior = torch.tensor([0.5, 0.15, 0.175, 0.175, 0.0], device=self.device)
        
    def save(self, fname: str) -> None:
        """Save a Q-network to a single ZIP file.

        Saved model cannot be used to re-training, as they only contain data essential
        for evaluation purpose.

        :param qnet_name: A file name for the Q-network states.
        """

        # Create a directory for the model checkpoints.
        basedir = os.path.dirname(fname)
        makedir_with_warning(basedir)

        # Create a dictionary containing the necessary parameters.
        # TODO: Provide an automatic method for populating the dictionaries
        qnet_params = {
            "num_hidden": self.num_hidden,
            "hidden_dim": self.hidden_dim,
        }

        with zipfile.ZipFile(fname, "w") as archive:
            with archive.open("params.pkl", "w") as f:
                pickle.dump(qnet_params, f)
            with archive.open("qnet.pt", "w") as f:
                torch.save(self.qnet_online.state_dict(), f)

    @classmethod
    def load(cls, fname: str, environment: PathfindEnv, device: str) -> Agent:
        """Load a class from a ZIP file.

        The loaded class should only be used for evaluation purposes. Please note that
        hyperparameters that are not essential for evaluation, such as discount factor,
        may not be reliable.

        :param fname: The path to the checkpoint file.
        :param env: The Gymnasium environment on which to evaluate the agent.
        :param device: A device to load the agent on.
        :returns: A class loaded from the ZIP file.
        """

        # Load parameters and state dictionaries from the ZIP file.
        with zipfile.ZipFile(fname, "r") as archive:
            with archive.open("params.pkl", "r") as f:
                params = pickle.load(f)
            with archive.open("qnet.pt", "r") as f:
                qnet_dict = torch.load(f)

        model = cls(
            environment=environment,
            num_hidden=params["num_hidden"],
            hidden_dim=params["hidden_dim"],
            device=device,
            # Dummy parameters (useless for the evaluation).
            seed=0,
        )
        model.qnet_online.load_state_dict(qnet_dict)
        model.qnet_online.eval()

        return model
    
    def learn(
        self,
        total_timesteps: int,
        checkpoint_dir: str | None = None,
        log_interval: int = 2000,
    ):
        """Train the agent.

        :param total_timestpes: The number of timesteps to train for.
        :param checkpoint_dir: The path to the directory for storing the checkpoint.
            If None, no checkpoint will be stored.
        """
        self.total_timesteps = total_timesteps
        pbar = tqdm(total = total_timesteps)
        t = 0
        t_logged = 0
        while t <= total_timesteps:
            len_episode, _, _ = self._rollout()
            t += len_episode
            pbar.update(len_episode)

            if self.buffer.size() >= self.batch_size:
                sample = self.buffer.sample(self.batch_size)
                self._train(*(dataclasses.astuple(sample) + (t,)))

            if t - t_logged >= log_interval:
                t_logged = t

                for city in list(City):
                    ep_len, ep_return, _ = self._rollout(options={"cities": [city]}, evaluation=True)
                    self.logger.log(
                        t,
                        {
                            f"{city}/ep_len": ep_len,
                            f"{city}/ep_return": ep_return,
                        },
                    )

                if checkpoint_dir is not None:
                    fname = f"checkpoint_{t}.zip"
                    self.save(os.path.join(checkpoint_dir, fname))
        self.environment.close()
        
    def _set_seed(self, seed: int):
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)
        self.environment.reset(seed=seed)  # It may look weird, but correct.
        
    @torch.no_grad()
    def _rollout(
        self, options: dict[str, Any] | None = None, evaluation: bool = False, start_id: PathfindID | None = None,
    ) -> tuple[int, float, tuple]:
        t = 0
        ep_return = 0.0
        obs, info = self.environment.reset(options=options, start_id=start_id)
        obs = self._concat_time(obs, info.city_time_remaining)
        pfid_list = []
        actions = []
        action_values = []
        similarities = []
        
        try:
            for t in range(self.per_city_time_limit):
                pfid_list.append(info.pf_id)
                action, action_value = self._select_action(obs, info.pf_id, evaluation=evaluation)
                actions.append(action)
                action_values.append(action_value)
                next_obs, reward, term, trunc, _, next_info = self.environment.step(action)
                next_obs = self._concat_time(next_obs, next_info.city_time_remaining)
                ep_return += 100.0 * float(reward)

                self.buffer.append(obs, action, 100.0 * float(reward), next_obs, term, info.city)

                # Augment the replay buffer by forcing 4th action.
                if not evaluation and action != 4:
                    # TODO: Remove _get_reward calling.
                    aug_reward = self.environment._get_reward(info.pf_id, 4)
                    aug_next_obs = self._concat_time(obs[:-1], next_info.city_time_remaining)
                    self.buffer.append(obs, 4, 100.0 * aug_reward, aug_next_obs, True, info.city)

                    # Update the exploration counter.
                    self.state_action_visitations[info.pf_id][4] += 1
                    
                if evaluation and action != 4:
                    similarity = self.environment._get_reward(info.pf_id, 4)
                    similarities.append(100.0 * similarity)
                elif evaluation and action == 4:
                    similarities.append(ep_return)

                if term or trunc:
                    break

                obs = next_obs
                info = next_info
           
        except TooManyRetriesError as e:

            raise e

        return t + 1, ep_return, (pfid_list, actions, action_values, similarities)
    
    @torch.no_grad()
    def _select_action(
        self,
        observation: npt.NDArray[np.float32],
        pathfind_id: PathfindID,
        evaluation: bool = False,
    ) -> int:
        tensor_observation = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        action_values = self.qnet_online(tensor_observation).squeeze()
        if evaluation:
            return int(torch.argmax(action_values)), action_values

        action_visitations = torch.tensor(
            self.state_action_visitations.get(pathfind_id, [0] * self.action_count),
            device=self.device,
        )
        exploration_bonus = (
            self.exploration_coefficient
            * self.exploration_prior
            * torch.sqrt(torch.sum(action_visitations))
            / (1 + action_visitations)
        )

        action = int(torch.argmax(action_values + exploration_bonus))
        action_visitations[action] += 1
        self.state_action_visitations[pathfind_id] = action_visitations.cpu().tolist()
        return action, action_values
    
    def _train(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        terminated: torch.Tensor,
        cities: torch.Tensor,
        step: int,
    ):
        # Predict a TD value.
        predicted_values = self.qnet_online(observations)
        action_values = torch.gather(predicted_values, 1, actions.unsqueeze(1))
        action_values = action_values.squeeze()

        # Calculate a TD target.
        with torch.no_grad():
            target_values = self.qnet_target(next_observations)
            next_action_values = self.qnet_online(next_observations)
            next_actions = torch.argmax(next_action_values, dim=-1, keepdim=True)

            target_values = torch.gather(target_values, 1, next_actions).squeeze()
            target_values = rewards + (1 - terminated) * (self.gamma * target_values)

        loss = self.crit(action_values, target_values)
        self.optimizer.zero_grad()
        clip_grad_norm_(self.qnet_online.parameters(), 10.0)
        loss.backward()

        self.optimizer.step()
        self._soft_update(self.qnet_online, self.qnet_target, self.qnet_tau)

        for city in list(City):
            idx = torch.from_numpy(cities == city)
            if not torch.any(idx):
                continue

            self.logger.log(
                step,
                {
                    f"{city}/reward_mean": torch.mean(rewards[idx]).item(),
                    f"{city}/reward_min": torch.min(rewards[idx]).item(),
                    f"{city}/reward_max": torch.max(rewards[idx]).item(),
                },
            )
            self.logger.log(
                step,
                {
                    f"{city}/target_mean": torch.mean(target_values[idx]).item(),
                    f"{city}/target_min": torch.min(target_values[idx]).item(),
                    f"{city}/target_max": torch.max(target_values[idx]).item(),
                },
            )
            self.logger.log(
                step,
                {
                    f"{city}/online_mean": torch.mean(action_values[idx]).item(),
                    f"{city}/online_min": torch.min(action_values[idx]).item(),
                    f"{city}/online_max": torch.max(action_values[idx]).item(),
                },
            )

        self.logger.log(
            step,
            {
                "loss": loss.item(),
            },
        )

    def _soft_update(self, local_model, target_model, tau):
        parameters = zip(target_model.parameters(), local_model.parameters())
        for t_param, l_param in parameters:
            t_param.data.copy_(tau * l_param.data + (1 - tau) * t_param.data)
            t_param.data.copy_(tau * l_param.data + (1 - tau) * t_param.data)

    def _concat_time(self, observation: npt.NDArray[np.float32], remaining_timestep: int):
        normalized_t = 2 * remaining_timestep / self.per_city_time_limit - 1
        return np.append(observation, normalized_t)