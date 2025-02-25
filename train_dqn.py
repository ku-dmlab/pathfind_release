import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

from agents.dqn import Agent
from env.common import City, TargetText
from env.online import PathfindEnv
from env.service import PathfindService
from utils.logger import (
    DummyLogger,
    StdandardOutputDecorator,
    TensorboardDecorator,
    WandBDecorator,
)

def parse_arguments(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--config-path", type=str, default="./config.json")

    # Hyperparameters for Logger
    parser.add_argument("--run-name", type=str, default="DQN")
    parser.add_argument("--logger", type=str, nargs="+", default=["wandb"])
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--no-print", action="store_false")

    # Hyperparameters for the PathfindEnv environment.
    parser.add_argument("--port", type=int, default=29000)
    parser.add_argument("--seed", type=int, default=1257)
    parser.add_argument("--city-spec", type=str, nargs="+", default=[[city.name] for city in City])
    parser.add_argument("--target-text", type=str, default=TargetText.THREE.name)

    # Hyperparameters for general training.
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--total-timesteps", type=int, default=300000)
    parser.add_argument("--gamma", type=float, default=0.999)

    # Hyperparameters for DQN model.
    parser.add_argument("--action-selector", type=str, default="eps")
    parser.add_argument("--num-hidden", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adam-betas", type=float, nargs=2, default=[0.9, 0.999])
    parser.add_argument("--adam-eps", type=float, default=1e-08)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--target-tau", type=float, default=0.005)

    return parser.parse_args(args)

def read_config(fname: str):
    with open(fname, "r") as f:
        config = json.load(f)
    return config


def suffix_timestamp(run_name: str) -> str:
    timestamp = datetime.fromtimestamp(time.time()).strftime("%m_%d_%H_%M_%S")
    return f"{run_name}_{timestamp}"

def main():
    args = parse_arguments(sys.argv[1:])

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    config = read_config(args.config_path)
    run_name = suffix_timestamp(args.run_name)

    logger = DummyLogger()
    if "stdout" in args.logger:
        logger = StdandardOutputDecorator(logger)
    if "tensorboard" in args.logger:
        logger = TensorboardDecorator(logger, log_directory=Path(args.output_dir) / "tensorboard")
    if "wandb" in args.logger:
        logger = WandBDecorator(logger, key=config["wandb_key"], project="pathfind", name=run_name)
    logger.log_hyperparameters(vars(args))
    with PathfindService(config["key"], server_address=("localhost", args.port)) as sv:
        if not isinstance(args.city_spec[0], list):
            city_spec = [[city] for city in args.city_spec]
        else:
            city_spec = args.city_spec

        env = PathfindEnv(
            api_key=config["key"],
            service=sv,
            city_spec=city_spec,
            text_spec=[args.target_text],
            capture_limit=1,
            device=device,
            logger=logger,
        )

        agent = Agent(
            seed=args.seed,
            environment=env,
            num_hidden=args.num_hidden,
            hidden_dim=args.hidden_dim,
            device=device,
            logger=logger,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            gamma=args.gamma,
            lr=args.lr,
            adam_betas=args.adam_betas,
            adam_eps=args.adam_eps,
            weight_decay=args.weight_decay,
            target_tau=args.target_tau,
        )

        checkpoint_dir = os.path.join(args.output_dir, run_name, "model")
        agent.learn(
            total_timesteps=args.total_timesteps,
            checkpoint_dir=checkpoint_dir,
            log_interval=10000,
        )


if __name__ == "__main__":
    main()
    