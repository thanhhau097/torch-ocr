import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output


class BidirectionalGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalGRU, self).__init__()

        self.rnn = nn.GRU(input_size, hidden_size, bidirectional=True)
        self.embedding = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        recurrent, hidden = self.rnn(x)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output, hidden


class BidirectionalGRU_2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pdropout = 0.3):
        super(BidirectionalGRU_2, self).__init__()

        self.rnn = nn.GRU(input_size, hidden_size, dropout=pdropout, num_layers=2, bidirectional=True)
        self.embedding = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        # input = [width, batch, channel * height]
        recurrent, _ = self.rnn(input)      # [width, batch, 2 x nHidden]
        output = self.embedding(recurrent)  # [width, batch, nOut]
        return output


if __name__ == '__main__':
    from ocr.data_loader.data_loaders import OCRDataLoader
    from ocr.data_loader.collate import collate_wrapper
    from ocr.model.visual_encoders.cnn_encoder import CNNEncoder

    dataloader = OCRDataLoader('../../data', 'train.json', 4, collate_fn=collate_wrapper)
    item = next(iter(dataloader))
    print('Input size:', item[0].size())

    encoder = CNNEncoder(3, 256)
    x = encoder(item[0])
    x = x.permute(3, 0, 2, 1)  # from B x C x H x W -> W x B x H x C
    size = x.size()
    x = x.reshape(size[0], size[1], size[2] * size[3])
    print('After CNN:', x.size())
    rnn_encoder = BidirectionalGRU(2048, 256, 256)
    x, hidden = rnn_encoder(x)
    print('After RNN:', x.size(), 'hidden:', hidden.size())
