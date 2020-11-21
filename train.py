import pytorch_lightning as pl

from pipelines import get_pipeline, get_callbacks
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

    logger = get_logger(args, cfg.logging)
    pipeline_cls = get_pipeline(cfg)
    model = pipeline_cls(experiment)
    callbacks, checkpoint_callback = get_callbacks(args, cfg)

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

    if args.tg_logging:
        tg_logger = experiment["tg_logger"]

        message = f"Experiment {args.checkpoints_dir}."
        if checkpoint_callback is not None:
            message += "\nBest {}: {:.4f}".format(
                checkpoint_callback.monitor,
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
