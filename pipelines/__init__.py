from pytorch_lightning import callbacks as cb

from utils import dict_remove_key
from .callbacks import *
from .multi_task_segmentator import MultiTaskSegmentator

available_pipelines = {
    "multi_task_segmentation": MultiTaskSegmentator,
}

available_callbacks = {
    "ModelCheckpoint": cb.ModelCheckpoint,
    "EarlyStopping": EarlyStopping,
    "GPUStatsMonitor": cb.GPUStatsMonitor,
    "GradientAccumulationScheduler": cb.GradientAccumulationScheduler,
    "LearningRateMonitor": cb.ModelCheckpoint,
    "FreezeEncoderCallback": FreezeEncoderCallback,
    "FreezeDecoderCallback": FreezeDecoderCallback,
    "Log2DSegmentationResultsCallback": Log2DSegmentationResultsCallback,
}


def get_pipeline(cfg):
    name = cfg.model.pipeline
    if name not in available_pipelines:
        raise KeyError(f"Pipeline {name} is not supported")

    return available_pipelines[name]


def get_callbacks(args, cfg):
    checkpoint_callback, callbacks_ = None, []
    for callback_cfg in cfg.callbacks:

        if callback_cfg.name == "ModelCheckpoint" and not args.debug:
            checkpoint_callback = callback = get_callback(
                callback_cfg,
                dirpath=args.checkpoints_dir,
                verbose=True,
            )

        else:
            callback = get_callback(callback_cfg)

        callbacks_ += [callback]

    return callbacks_, checkpoint_callback


def get_callback(callback_cfg, **kwargs):
    name = callback_cfg.name
    if name not in available_callbacks:
        raise KeyError(f"Callback {name} is not supported")

    callback_cls = available_callbacks[name]
    params = dict_remove_key(callback_cfg, "name")

    return callback_cls(**kwargs, **params)
