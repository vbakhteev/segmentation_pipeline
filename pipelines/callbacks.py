import numpy as np
import torch
from pytorch_lightning import callbacks as cb


class EarlyStopping(cb.EarlyStopping):
    def teardown(self, trainer, pl_module, stage):
        torch_inf = torch.tensor(np.Inf)
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf
