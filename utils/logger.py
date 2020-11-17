from pytorch_lightning.loggers.comet import CometLogger
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers.mlflow import MLFlowLogger
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.test_tube import TestTubeLogger
from pytorch_lightning.loggers.wandb import WandbLogger

available_loggers = dict(
    csv_logs=CSVLogger,
    mlflow=MLFlowLogger,
    neptune=NeptuneLogger,
    comet=CometLogger,
    tensorboard=TensorBoardLogger,
    test_tube=TestTubeLogger,
    wandb=WandbLogger,
)


def get_logger(args, cfg_logging):
    # TODO add specific to experiment name
    if cfg_logging.name is None:
        return None
    if cfg_logging.name not in available_loggers:
        raise KeyError(f"Logger {cfg_logging.name} is not supported.")

    logger_cls = available_loggers[cfg_logging.name]
    logger = logger_cls(**cfg_logging.params)
    return logger
