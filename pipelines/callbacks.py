import numpy as np
import torch
from pytorch_lightning import callbacks as cb

from models.utils import freeze, unfreeze


class EarlyStopping(cb.EarlyStopping):
    def teardown(self, trainer, pl_module, stage):
        torch_inf = torch.tensor(np.Inf)
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf


class FreezeEncoderCallback(cb.Callback):
    def __init__(self, epochs=1):
        self.epochs = epochs

    def on_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch < self.epochs:
            freeze(pl_module.model.model.encoder)
        else:
            unfreeze(pl_module.model.model.encoder)
            self.epochs = 0     # To not freeze model in next stages


class FreezeDecoderCallback(cb.Callback):
    def __init__(self, epochs=1):
        self.epochs = epochs

    def on_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch < self.epochs:
            freeze(pl_module.model.model.decoder)
        else:
            unfreeze(pl_module.model.model.decoder)
            self.epochs = 0
