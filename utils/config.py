import argparse
import copy
import os
import shutil
from datetime import datetime
from pathlib import Path

import yaml
from easydict import EasyDict

from .stuff import set_deterministic
from .tg_logger import TelegramLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline")
    parser.add_argument("config_file", type=str, help="path to config file")
    parser.add_argument(
        "--checkpoints_dir", type=str, default=None, help="models are saved here"
    )
    parser.add_argument(
        "--no_tg",
        action="store_true",
        help="send train information to telegram after training",
    )

    parser.add_argument(
        "--debug", action="store_true", help="Turns off logging and checkpointing"
    )
    return parser.parse_args()


def setup_experiment():
    args = parse_args()
    cfg = load_config(args.config_file)
    experiment = dict(args=args, cfg=cfg)

    set_deterministic(cfg.defaults.seed)

    if args.debug:
        args.no_tg = True

    if not args.no_tg:
        tg_logger = TelegramLogger()
        experiment["tg_logger"] = tg_logger

    if args.checkpoints_dir is None:
        checkpoints_name = extract_config_name(args.config_file)
        checkpoints_name = checkpoints_name + datetime.now().strftime("_%B%d_%H:%M:%S")
        args.checkpoints_dir = "checkpoints/" + checkpoints_name

    checkpoints_dir = Path(args.checkpoints_dir)
    if not args.debug:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.config_file, checkpoints_dir / "config.yml")

    return experiment


def load_config(config_path):
    with open(config_path, "r") as fid:
        config = yaml.load(fid, Loader=yaml.FullLoader)
    config = EasyDict(config)
    return config


def extract_config_name(config_path):
    config_name = os.path.split(config_path)[-1]
    return config_name.replace(".yml", "")


def update_config(cfg, stage_cfg):
    cfg = copy.deepcopy(cfg)

    for component_name, component_cfg in stage_cfg.items():

        if component_name in ("lightning", "dataloader"):
            # Update changes
            for k, v in component_cfg.items():
                cfg[component_name][k] = v

        elif component_name == "datasets":
            for dataset_cfg in component_cfg.list:

                # Find index of dataset which will be changed
                idx_dataset_to_replace = -1
                for i in range(len(cfg.datasets)):
                    if cfg.datasets.list[i].id == dataset_cfg.id:
                        idx_dataset_to_replace = i
                        break

                # Update components of dataset_cfg
                for k, v in dataset_cfg.items():
                    cfg.datasets.list[idx_dataset_to_replace][k] = v

        elif component_name == "model":
            # model.{task}.heads[target_name].criterion
            for task_name, stage_task_cfg in component_cfg.items():
                for stage_head_cfg in stage_task_cfg.heads:

                    target_name = stage_head_cfg.target

                    # Find index of head
                    idx_head = -1
                    for i in range(len(cfg.model[task_name].heads)):
                        if cfg.model[task_name].heads[i].target == target_name:
                            idx_head = i
                            break

                    # Replace criterion
                    cfg.model[task_name].heads[
                        idx_head
                    ].criterion = stage_head_cfg.criterion

        elif component_name in ("optimizer", "scheduler"):
            # Replace original component with new
            cfg[component_name] = copy.deepcopy(component_cfg)

        else:
            raise KeyError(
                f"Component {component_name} can't be replaced during training."
            )

    return cfg
