from .base_single_model_pipeline import BaseSingleModel
from models import get_model


class Segmentator(BaseSingleModel):
    def __init__(self, experiment):
        super().__init__(experiment)

        self.model = get_model(
            model_type="segmentation",
            name=self.cfg.model.name,
            params=self.cfg.model.params,
            n_dim=self.cfg.model.n_dim,
        )

    def training_step(self, batch, batch_idx):
        mask = batch["mask"]
        logits_mask = self.model(batch)
        loss = self.criterion(logits_mask, mask)

        return loss

    def validation_step(self, batch, batch_idx):
        mask = batch["mask"]
        logits_mask = self.model(batch)

        loss = self.criterion(logits_mask, mask)
        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        for metric_name in self.logging_names:
            metric_o = getattr(self, metric_name)
            score = metric_o(logits_mask, mask)
            self.log(metric_name, score, on_step=False, on_epoch=True, prog_bar=True)
