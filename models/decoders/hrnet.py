from .base_decoder import BaseDecoder, EncoderDecoder

hr_nets = (
    "hrnet_w18_small",
    "hrnet_w18_small_v2",
    "hrnet_w18",
    "hrnet_w30",
    "hrnet_w32",
    "hrnet_w40",
    "hrnet_w44",
    "hrnet_w48",
    "hrnet_w64",
)


class HRNet(EncoderDecoder):
    def __init__(
        self,
        encoder_name: str,
        n_dim: int,
        in_channels: int,
    ):
        if encoder_name not in hr_nets:
            raise KeyError(
                f"To use HRNet you need to pass `encoder_name` one of: {hr_nets}"
            )

        super().__init__(encoder_name, n_dim, in_channels)
        self.decoder = HRNetDecoder(out_channels=self.encoder.out_channels[0])


class HRNetDecoder(BaseDecoder):
    def __init__(self, out_channels):
        super().__init__(out_channels)

    def forward(self, *features):
        return features[0]
