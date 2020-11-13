import pytorch_lightning as pl

from models import get_pipeline
from utils import dict_remove_key, setup_experiment, update_config


def main():
    experiment = setup_experiment()
    args, cfg = experiment["args"], experiment["cfg"]

    pipeline_cls = get_pipeline(cfg)
    model = pipeline_cls(experiment)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.checkpoints_dir,
        verbose=True,
        monitor="val_metric",
        **dict_remove_key(cfg.checkpointing, "metric"),
    )

    for stage_name, stage_cfg in cfg.train_stages.items():
        print(f"Start stage: {stage_name}\n" + "-" * 40)

        cfg = update_config(cfg, stage_cfg)
        model.update_config(cfg)

        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            weights_summary=None,
            progress_bar_refresh_rate=1,
            # overfit_pct=train_sample,
            num_sanity_val_steps=20,
            **cfg.lightning,
        )
        trainer.fit(model)

    if args.tg_logging:
        tg_logger = experiment["tg_logger"]
        tg_logger.send_message("Best Validation Metric: ")
        # tg_logger.send_image()


if __name__ == "__main__":
    main()
