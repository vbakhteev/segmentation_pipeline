import pytorch_lightning as pl
import torch

from pipelines import get_pipeline, get_callbacks
from utils import (
    dict_remove_key,
    setup_experiment,
    update_config,
    get_logger,
    log_to_tg,
    TelegramLogger,
)


def main():
    experiment = setup_experiment()
    args, cfg = experiment["args"], experiment["cfg"]

    logger = get_logger(args, cfg.logging)
    model = get_pipeline(cfg)
    callbacks, checkpoint_callback, weight_ensemble_callback = get_callbacks(args, cfg)

    for stage_cfg in cfg.train_stages:
        stage_name = stage_cfg.name
        print("#" * 40 + f"\nStart stage: {stage_name}\n" + "#" * 40)

        cfg = update_config(cfg, dict_remove_key(stage_cfg, "name"))
        model.update_config(cfg)

        trainer = pl.Trainer(
            callbacks=callbacks,
            logger=logger,
            weights_summary=None,
            progress_bar_refresh_rate=1,
            num_sanity_val_steps=0,
            **cfg.lightning,
        )
        trainer.fit(model)

    # Validate ensemble of weights
    if weight_ensemble_callback is not None:
        # Don't update parameters
        model.cfg.optimizer[0].lr = 1e-15

        trainer = pl.Trainer(
            callbacks=callbacks,
            logger=logger,
            weights_summary=None,
            progress_bar_refresh_rate=1,
            num_sanity_val_steps=0,
            gpus=cfg.lightning.get("gpus", 0),
            max_epochs=1,
        )
        trainer.fit(model)

    if checkpoint_callback is not None:
        print("Restore best checkpoint")
        model_path = checkpoint_callback.best_model_path
        state_dict = torch.load(model_path)["state_dict"]
        model.load_state_dict(state_dict)

    log_to_tg(model, experiment, checkpoint_callback)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        tg_logger = TelegramLogger()
        tg_logger.send_message(str(e))
        raise e
