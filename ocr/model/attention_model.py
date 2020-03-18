import torch
import torch.nn as nn
from ..base import BaseModel
from ..data_loader.vocab import SOS_token
from ..model.decoders.attention_decoder import LuongAttnDecoderRNN
from ..model.visual_encoders.cnn_encoder import CNNEncoder

from ..model.rnn_encoders.rnn_encoder import BidirectionalGRU


class AttentionModel(BaseModel):
    def __init__(self, num_chars):
        super().__init__()
        out_dimension = 256
        self.encoder = CNNEncoder(3, out_dimension)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # replace downsample factor with AvgPool
        self.rnn_encoder = BidirectionalGRU(out_dimension, 256, 256)  # change here
        self.num_chars = num_chars
        embedding = nn.Embedding(num_chars, 256)
        self.decoder = LuongAttnDecoderRNN('general', embedding, 256, num_chars)

    def forward(self, x, labels, max_label_length, device, training=True):
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

        # --------------- ATTENTION DECODER -------------------
        batch_size = encoder_outputs.size()[1]
        decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
        decoder_input = decoder_input.to(device)
        decoder_hidden = last_hidden[:self.decoder.n_layers]

        # Forward batch of sequences through decoder one time step at a time
        teacher_forcing_ratio = 0.5  # TODO: change to dynamic teacher forcing
        use_teacher_forcing = True  # if random.random() < teacher_forcing_ratio else False

        outputs = torch.zeros((max_label_length, batch_size, self.num_chars), device=device)

        # print("Get device", decoder_input.get_device(), decoder_hidden.get_device(), encoder_outputs.get_device())

        if use_teacher_forcing and training:
            for t in range(max_label_length):
                # print('timestep:', t)
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # Teacher forcing: next input is current target
                decoder_input = labels[t].view(1, -1)
                outputs[t] = decoder_output  # batch_size * num_chars
                # print(decoder_input.get_device())
        else:
            for t in range(max_label_length):
                # print('timestep:', t)
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
                decoder_input = decoder_input.to(device)
                outputs[t] = decoder_output

        return outputs


if __name__ == '__main__':
    from ..data_loader.data_loaders import OCRDataLoader
    from ..data_loader.collate import collate_wrapper

    dataloader = OCRDataLoader('../data', 'train.json', 4, collate_fn=collate_wrapper)
    item = next(iter(dataloader))
    print('Input size:', item[0].size())

    device = torch.device("cpu")
    model = AttentionModel(num_chars=65)
    x = model(item[0], item[1], item[3], device)
    print("After Decoder", x.size())
