import pytorch_lightning as pl

from pipelines import get_pipeline
from utils import (
    dict_remove_key,
    setup_experiment,
    update_config,
    get_logger,
    metrics_to_image,
)


def main():
    experiment = setup_experiment()
    args, cfg = experiment["args"], experiment["cfg"]

    pipeline_cls = get_pipeline(cfg)
    model = pipeline_cls(experiment)

    callbacks = []
    if not args.debug:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=args.checkpoints_dir,
            verbose=True,
            monitor="val_metric",
            **dict_remove_key(cfg.checkpointing, "metric"),
        )
        callbacks += [checkpoint_callback]
    logger = get_logger(args, cfg.logging)

    for stage_cfg in cfg.train_stages:
        stage_name = stage_cfg.name
        print(f"Start stage: {stage_name}\n" + "-" * 40)

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

    if args.tg_logging:
        tg_logger = experiment["tg_logger"]
        message = "Experiment {}.\nBest validation {}: {:.4f}".format(
            args.checkpoints_dir,
            cfg.checkpointing.metric.name,
            checkpoint_callback.best_model_score,
        )
        tg_logger.send_message(message)

        # TODO get validation loss
        metrics = model.logged_metrics
        if len(metrics):
            img = metrics_to_image(metrics)
            tg_logger.send_image(img)


if __name__ == "__main__":
    main()
