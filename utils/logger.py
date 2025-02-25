import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torchvision.transforms.functional as F
from numpy import typing as npt
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter

import wandb
from utils.file import makedir_with_warning


# TODO: Create a config module.
def read_config(fname: str) -> dict:
    with open(fname, "r") as f:
        config = json.load(f)
    return config


class Image:
    def __init__(self, image: Tensor | npt.NDArray[np.float32]):
        self.image = image


# TODO: Add a class for Hyperparameters and Metrics.


class Logger(ABC):
    @abstractmethod
    def log(self, time_step: int, key_values: dict[str, Any]) -> None:
        raise NotImplementedError()

    @abstractmethod
    def log_hyperparameters(self, hyperparameters: dict[str, Any]) -> None:
        raise NotImplementedError


class DummyLogger(Logger):
    """Implements the Logger interface, while not logging actual data.

    This class serves as a replacement of 'None' value for the logger, and will be
    used as a default logger for many cases.
    """

    def log(self, time_step: int, key_values: dict[str, Any]) -> None:
        pass

    def log_hyperparameters(self, hyperparameters: dict[str, Any]) -> None:
        pass


class Decorator(Logger):
    def __init__(self, logger: Logger) -> None:
        self._logger = logger

    def log(self, time_step: int, key_values: dict[str, Any]) -> None:
        self._logger.log(time_step, key_values)

    def log_hyperparameters(self, hyperparameters: dict[str, Any]) -> None:
        self._logger.log_hyperparameters(hyperparameters)

    @property
    def logger(self) -> Logger:
        return self._logger


class StdandardOutputDecorator(Decorator):
    def log(self, time_step: int, key_values: dict[str, Any], prefix: str = "") -> None:
        super().log(time_step, key_values)
        for key, value in key_values.items():
            if isinstance(value, Image):
                print(f"{time_step}_{prefix}{key}: an image.")
            elif isinstance(value, dict):
                self.log(time_step, value, prefix=f"{key}/")
            else:
                print(f"{time_step}_{prefix}{key}: {value}")

    def log_hyperparameters(self, hyperparameters: dict[str, Any]) -> None:
        super().log_hyperparameters(hyperparameters)
        print("\n".join([f"|{key}|{value}" for key, value in hyperparameters.items()]))


class TensorboardDecorator(Decorator):
    def __init__(self, logger: Logger, log_directory: Path):
        super().__init__(logger)

        makedir_with_warning(log_directory)
        self.writer = SummaryWriter(log_dir=log_directory)

    def log(self, time_step: int, key_values: dict[str, Any], prefix: str = "") -> None:
        for key, value in key_values.items():
            if isinstance(key, np.ScalarType):
                if isinstance(value, str):
                    self.writer.add_text(f"{prefix}{key}", value, global_step=time_step)
                else:
                    self.writer.add_scalar(f"{prefix}{key}", value, global_step=time_step)
            elif isinstance(value, Image):
                self.writer.add_image(key, value.image, global_step=time_step)
            elif isinstance(value, dict):
                self.log(time_step, value, prefix=f"{key}/")
            else:
                logging.error(
                    f"{value} with {prefix}{key} at {time_step} is not of supported type."
                )

    def log_hp(self, hyperparameters: dict[str, Any]) -> None:
        # TODO: Consider using self.writer.add_hparams or add_summary.
        self.writer.add_text(
            "hyperparatmers",
            "\n".join([f"|{key}|{value}" for key, value in hyperparameters.items()]),
        )


class WandBDecorator(Decorator):
    """Log the data using Weight & Biases (W&B) API.

    Instead of passing 'step' parameter to wandb.log(step=step, ...), the WandBLogger
    logs it as a separate timestep value. This approach addresses two issues: 1) When
    multiple metrics are logged at different frequencies, W&B often displays an almost
    uninterpretable iteration value, and 2) using step=step in wandb.log necessitates
    maintaining a coherent global timestep across different classes and methods, which
    can be challenging.

    Please note that this solution is recommended in their official examples.
    """

    def __init__(self, logger: Logger, key: str | None, project: str | None, name: str | None):
        super().__init__(logger)

        if key is not None:
            wandb.login(key=key)
        wandb.init(project=project, name=name)

    def log(self, time_step: int, key_values: dict[str, Any]):
        super().log(time_step, key_values)

        data = self._sanitize_data(key_values)
        data["time_step"] = time_step
        wandb.log(data)

    def log_hyperparameters(self, hyperparameters: dict[str, Any]):
        super().log_hyperparameters(hyperparameters)
        wandb.config.update(hyperparameters)

    def _sanitize_data(self, key_values: dict[str, Any]) -> dict[str, Any]:
        wandb_data = {}
        for key, value in key_values.items():
            if isinstance(value, Image):
                wandb_data[key] = wandb.Image(value.image)
            elif isinstance(value, dict):
                self._sanitize_data(value)
            else:
                wandb_data[key] = value
        return wandb_data
