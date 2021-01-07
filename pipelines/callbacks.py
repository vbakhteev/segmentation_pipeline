import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from timm.utils.model_ema import ModelEmaV2
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
            self.epochs = 0  # To not freeze model in next stages


class FreezeDecoderCallback(cb.Callback):
    def __init__(
        self,
        epochs=1,
    ):
        self.epochs = epochs

    def on_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch < self.epochs:
            freeze(pl_module.model.model.decoder)
        else:
            unfreeze(pl_module.model.model.decoder)
            self.epochs = 0


class EMACallback(cb.Callback):
    def __init__(self, decay, update_nth_iter=5, device=None):
        """
        Args:
        update_nth_iter: Weights updates at each nth train iteration.
        decay: Decay of model's weights.
        device: Where to store weights. If None then on the same device.

        Hint: decay should be such that decay^(steps_per_epoch / update_nth_iter) ~ [0.3, 0.7]
        If epoch is long then better to expression=0.3
        If epoch is short then better to set expression=0.7
        """
        self.update_nth_iter = update_nth_iter
        self.decay = decay
        self.device = device
        self.initialized = False
        self.weights = None

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if (batch_idx % self.update_nth_iter) != 0:
            return

        if self.initialized:
            self.weights.update(pl_module.model)
        else:
            self.weights = ModelEmaV2(
                pl_module.model, decay=self.decay, device=self.device
            )
            self.initialized = True

    def on_train_end(self, trainer, pl_module):
        ema_state_dict = self.weights.module.state_dict()
        pl_module.model.load_state_dict(ema_state_dict)


class SWACallback(cb.Callback):
    def __init__(self, start_epoch, device=None):
        """
        Args:
        start_epoch: From which epoch to collect weights.
        device: Where to store weights. If None then on the same device.
        """
        self.start_epoch = start_epoch
        self.device = device
        self.epoch = 0
        self.initialized = False
        self.weights = None

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        if self.start_epoch > self.epoch:
            self.epoch += 1
            return

        if self.initialized:
            self.weights.update_parameters(pl_module.model)
        else:
            self.weights = torch.optim.swa_utils.AveragedModel(
                pl_module.model, device=self.device
            )
            self.weights.update_parameters(pl_module.model)
            self.initialized = True

        self.epoch += 1

    def on_train_end(self, trainer, pl_module):
        if not self.initialized:
            return

        swa_state_dict = self.weights.module.state_dict()
        pl_module.model.load_state_dict(swa_state_dict)

        # At the beginning of new stage use new weights
        self.weights = None
        self.initialized = False


class Log2DSegmentationResultsCallback(cb.Callback):
    def __init__(self, target_mask: str, n_images=1, batch_idx=0):
        """Logs images and model's segmentation masks to your favourite Logger.
        target_mask: the name of target mask.
        n_images: how many images to log.
        batch_idx: from which batch take images.
        """
        super().__init__()
        self.target_mask = target_mask
        self.n_images = n_images
        self.batch_idx = batch_idx
        self.current_epoch = 0

    def on_validation_batch_end(
        self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if batch_idx != self.batch_idx:  # Log only specific batch.
            return

        images = batch["image"]
        gts = batch[self.target_mask]
        masks = pl_module.last_model_outputs[self.target_mask]

        images -= images.min()
        images /= images.max()
        images = images.cpu().permute(0, 2, 3, 1)
        gts = gts.cpu()
        if masks.shape[1] == 2:
            masks = torch.softmax(masks, dim=1)[:, 1, ...]
        else:
            masks = masks.argmax(1)
        masks = masks.cpu()

        self._plot(images, masks, gts, trainer)
        self.current_epoch += 1

    def _plot(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        gts: torch.Tensor,
        trainer: pl.Trainer,
    ) -> None:
        for img_i, (image, mask, gt) in enumerate(zip(images, masks, gts)):
            if img_i >= self.n_images:
                return

            fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(15, 10))
            self.__draw_sample(fig, axarr, 0, image, "Image")
            self.__draw_sample(fig, axarr, 1, mask, "Mask")
            self.__draw_sample(fig, axarr, 2, gt, "Ground Truth")

            trainer.logger.experiment.add_figure(
                "segmentation result", fig, global_step=self.current_epoch
            )

    @staticmethod
    def __draw_sample(
        fig: plt.Figure, axarr: plt.Axes, col_idx: int, img: torch.Tensor, title: str
    ):
        im = axarr[col_idx].imshow(img)
        fig.colorbar(im, ax=axarr[col_idx])
        axarr[col_idx].set_title(title, fontsize=20)
