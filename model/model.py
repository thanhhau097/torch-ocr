import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from base import BaseModel
from .visual_encoders.cnn_encoder import CNNEncoder  # TODO: bug?
from .rnn_encoders.rnn_encoder import BidirectionalGRU
from model.decoders.attention_decoder import LuongAttnDecoderRNN
from data_loader.vocab import SOS_token


class OCRModel(BaseModel):
    def __init__(self, num_chars=65):
        super().__init__()
        self.encoder = CNNEncoder(3, 256)
        self.rnn_encoder = BidirectionalGRU(2048, 256, 256)
        embedding = nn.Embedding(num_chars, 256)
        self.decoder = LuongAttnDecoderRNN('general', embedding, 256, num_chars)

    def forward(self, x, labels, max_label_length):
        # ---------------- CNN ENCODER --------------
        x = self.encoder(x)
        print('After CNN:', x.size())

        # ---------------- CNN TO RNN ----------------
        x = x.permute(3, 0, 2, 1)  # from B x C x H x W -> W x B x H x C
        size = x.size()
        x = x.reshape(size[0], size[1], size[2] * size[3])

        # ----------------- RNN ENCODER ---------------
        encoder_outputs, last_hidden = self.rnn_encoder(x)
        print('After RNN', x.size())

        # --------------- ATTENTION DECODER -------------------
        batch_size = encoder_outputs.size()[1]
        decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
        decoder_hidden = last_hidden[:self.decoder.n_layers]

        # Forward batch of sequences through decoder one time step at a time
        teacher_forcing_ratio = 0.5
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        # TODO: we need to calculate closs in trainer.py file
        if use_teacher_forcing:
            for t in range(max_label_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # Teacher forcing: next input is current target
                decoder_input = labels[t].view(1, -1)
        else:
            for t in range(max_label_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
                # print("Decoder Output:", decoder_output.size())
                # print("Decoder Input:", decoder_input)

            # print("After Decoder", decoder_output.size())
            return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    from data_loader.data_loaders import OCRDataLoader
    from data_loader.collate import collate_wrapper
    from model.visual_encoders.cnn_encoder import CNNEncoder
    from model.rnn_encoders.rnn_encoder import BidirectionalGRU
    from data_loader.vocab import SOS_token

    dataloader = OCRDataLoader('../../data', 'train.json', 4, collate_fn=collate_wrapper)
    item = next(iter(dataloader))
    print('Input size:', item[0].size())

    model = OCRModel(num_chars=65)
    x = model(item[0], item[1], item[3])
    print(x.size())