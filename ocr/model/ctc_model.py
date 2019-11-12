import torch
import torch.nn as nn
from ocr.base import BaseModel
from ocr.model.visual_encoders.cnn_encoder import CNNEncoder

from ocr.model.rnn_encoders.rnn_encoder import BidirectionalGRU


class CTCModel(BaseModel):
    def __init__(self, num_chars=65):
        super().__init__()
        self.encoder = CNNEncoder(3, 512)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
        self.rnn_encoder = BidirectionalGRU(512, 256, 256)
        self.num_chars = num_chars
        self.decoder = nn.Linear(256, self.num_chars)

    def forward(self, x, labels=None, max_label_length=None, device=None, training=True):
        # ---------------- CNN ENCODER --------------
        x = self.encoder(x)
        # print('After CNN:', x.size())

        # ---------------- CNN TO RNN ----------------
        x = x.permute(3, 0, 1, 2)  # from B x C x H x W -> W x B x C x H
        x = self.AdaptiveAvgPool(x)
        size = x.size()
        x = x.reshape(size[0], size[1], size[2] * size[3])

        # ----------------- RNN ENCODER ---------------
        encoder_outputs, last_hidden = self.rnn_encoder(x)
        # print('After RNN', x.size())

        # --------------- CTC DECODER -------------------
        # batch_size = encoder_outputs.size()[1]
        outputs = self.decoder(encoder_outputs)

        return outputs


if __name__ == '__main__':
    from ocr.data_loader.data_loaders import OCRDataLoader
    from ocr.data_loader.collate import collate_wrapper

    dataloader = OCRDataLoader('../data', 'train.json', 4, collate_fn=collate_wrapper)
    item = next(iter(dataloader))
    print('Input size:', item[0].size())

    device = torch.device("cpu")
    model = CTCModel(num_chars=65)
    x = model(item[0], item[1], item[3], device)
    print("After Decoder", x.size())
