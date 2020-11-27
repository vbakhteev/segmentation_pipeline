from torch import nn


class BaseDecoder(nn.Module):
    def __init__(self):
        super().__init__()


class SegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = None
        self.decoder = None
        self.segmentation_head = None
        self.classification_head = None

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks
