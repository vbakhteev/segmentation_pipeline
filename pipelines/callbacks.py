from pytorch_lightning import callbacks as cb
from utils import dict_remove_key

available_callbacks = {
    "ModelCheckpoint": cb.ModelCheckpoint,
    "EarlyStopping": cb.EarlyStopping,
    "GPUStatsMonitor": cb.GPUStatsMonitor,
    "GradientAccumulationScheduler": cb.GradientAccumulationScheduler,
    "LearningRateMonitor": cb.ModelCheckpoint,
}


def get_callbacks(args, cfg):
    checkpoint_callback, callbacks = None, []
    for callback_cfg in cfg.callbacks:

        if callback_cfg.name == "ModelCheckpoint":
            checkpoint_callback = callback = get_callback(
                callback_cfg,
                dirpath=args.checkpoints_dir,
                verbose=True,
            )
        else:
            callback = get_callback(callback_cfg)

        callbacks += [callback]

    return callbacks, checkpoint_callback


def get_callback(callback_cfg, **kwargs):
    name = callback_cfg.name
    if name not in available_callbacks:
        raise KeyError(f"Callback {name} is not supported")

    callback_cls = available_callbacks[name]
    params = dict_remove_key(callback_cfg, "name")

    return callback_cls(**kwargs, **params)
