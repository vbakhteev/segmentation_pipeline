from torch import nn

from .modules import get_segmentation_head, get_classification_head
from .decoders.base_decoder import EncoderDecoder
from .utils import initialize_decoder, initialize_head


class MultiHeadSegmentator(nn.Module):
    def __init__(
        self,
        model: EncoderDecoder,
        n_dim: int,
        seg_heads_cfgs: list,
        clf_heads_cfgs: list,
    ):
        super().__init__()

        self.model = model
        encoder_out_channels = self.model.encoder.out_channels[-1]
        decoder_out_channels = self.model.decoder.out_channels

        self.clf_heads = nn.ModuleDict()
        for head_cfg in clf_heads_cfgs:
            self.clf_heads[head_cfg.target] = get_classification_head(
                head=head_cfg.head,
                in_channels=encoder_out_channels,
                n_dim=n_dim,
                **head_cfg.params,
            )

        self.seg_heads = nn.ModuleDict()
        for head_cfg in seg_heads_cfgs:
            self.seg_heads[head_cfg.target] = get_segmentation_head(
                head=head_cfg.head,
                in_channels=decoder_out_channels,
                n_dim=n_dim,
                **head_cfg.params,
            )

        self.initialize()

    def initialize(self):
        initialize_decoder(self.model.decoder)
        initialize_head(self.seg_heads)
        initialize_head(self.clf_heads)

    def forward(self, images):
        encoder_features, decoder_output = self.model(images)

        outputs = dict()
        for target, seg_head in self.seg_heads.items():
            outputs[target] = seg_head(decoder_output)

        for target, clf_head in self.clf_heads.items():
            outputs[target] = clf_head(encoder_features[-1])

        return outputs
