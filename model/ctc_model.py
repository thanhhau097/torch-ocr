import torch
import torch.nn as nn
import random

from base import BaseModel
from model.visual_encoders.cnn_encoder import CNNEncoder
from model.rnn_encoders.rnn_encoder import BidirectionalGRU
from model.decoders.attention_decoder import LuongAttnDecoderRNN
from data_loader.vocab import SOS_token


class CTCModel(BaseModel):
    def __init__(self, num_chars=65):
        super().__init__()
        self.encoder = CNNEncoder(3, 256)
        # in_dimension = height / self.encoder.downsample_factor * 256  # TODO: pass height
        self.rnn_encoder = BidirectionalGRU(2048, 256, 256)
        self.num_chars = num_chars
        self.decoder = nn.Linear(256, self.num_chars)

    def forward(self, x, labels, max_label_length, device, training=True):
        # ---------------- CNN ENCODER --------------
        x = self.encoder(x)
        # print('After CNN:', x.size())

        # ---------------- CNN TO RNN ----------------
        x = x.permute(3, 0, 2, 1)  # from B x C x H x W -> W x B x H x C
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
    from data_loader.data_loaders import OCRDataLoader
    from data_loader.collate import collate_wrapper

    dataloader = OCRDataLoader('../data', 'train.json', 4, collate_fn=collate_wrapper)
    item = next(iter(dataloader))
    print('Input size:', item[0].size())

    device = torch.device("cpu")
    model = CTCModel(num_chars=65)
    x = model(item[0], item[1], item[3], device)
    print("After Decoder", x.size())
