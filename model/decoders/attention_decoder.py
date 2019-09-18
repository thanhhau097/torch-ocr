import torch
from torch import nn
import torch.nn.functional as F

from model.decoders.attention import Attn


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden


if __name__ == '__main__':
    from data_loader.data_loaders import OCRDataLoader
    from data_loader.collate import collate_wrapper
    from model.visual_encoders.cnn_encoder import CNNEncoder
    from model.rnn_encoders.rnn_encoder import BidirectionalGRU
    from data_loader.vocab import SOS_token

    dataloader = OCRDataLoader('../../data', 'train.json', 4, collate_fn=collate_wrapper)
    item = next(iter(dataloader))
    print('Input size:', item[0].size())

    encoder = CNNEncoder(3, 256)
    x = encoder(item[0])
    print('After CNN:', x.size())
    x = x.permute(3, 0, 2, 1)  # from B x C x H x W -> W x B x H x C
    size = x.size()
    x = x.reshape(size[0], size[1], size[2] * size[3])

    # ----------------- RNN ENCODER ---------------
    rnn_encoder = BidirectionalGRU(2048, 256, 256)
    encoder_outputs, last_hidden = rnn_encoder(x)
    print('After RNN', x.size())

    # --------------- ATTENTION DECODER -------------------
    voc = dataloader.get_vocab()
    print('Num chars:', voc.num_chars)
    embedding = nn.Embedding(voc.num_chars, 256)
    decoder = LuongAttnDecoderRNN('general', embedding, 256, voc.num_chars)

    batch_size = encoder_outputs.size()[1]
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_hidden = last_hidden[:decoder.n_layers]

    max_label_length = item[3]
    for t in range(max_label_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        # No teacher forcing: next input is decoder's own current output
        _, topi = decoder_output.topk(1)
        decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
        print("Decoder Output:", decoder_output.size())
        print("Decoder Input:", decoder_input)

    # print("After Decoder", decoder_output.size())
