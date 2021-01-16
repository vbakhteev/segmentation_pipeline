import argparse

import pytorch_lightning as pl

from pipelines import get_pipeline
from models.utils import load_state_dict
from utils import dict_remove_key
from utils.config import load_config, update_config


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline")
    parser.add_argument("config_file", type=str, help="path to config file")
    parser.add_argument("checkpoint_path", type=str, help="path to model's weights")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config_file)

    model = get_pipeline(cfg)
    load_state_dict(model.model, args.checkpoint_path, soft=True)

    for stage_cfg in cfg.train_stages:
        cfg = update_config(cfg, dict_remove_key(stage_cfg, "name"))
    model.update_config(cfg)

    trainer = pl.Trainer(
        weights_summary=None,
        progress_bar_refresh_rate=1,
        **cfg.lightning,
    )
    trainer.test(model)


if __name__ == '__main__':
    main()
