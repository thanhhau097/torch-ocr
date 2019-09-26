import torch
from model.ctc_model import CTCModel
from data_loader.vocab import Vocab
import numpy as np
from data_loader.collate import process_image
import itertools


class LionelOCR():
    def __init__(self, weights_path, vocab_path):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.voc = Vocab()
        self.voc.build_vocab_from_char_dict_file(vocab_path)
        self.model = CTCModel(self.voc.num_chars)
        self.model = self.model.to(self.device)

        # # LOAD MODEL
        # checkpoint = torch.load(weights_path)
        # state_dict = checkpoint['state_dict']
        # model.load_state_dict(state_dict)

    def process(self, image):
        self.model.eval()
        # preprocess image, TODO: height = 64 (no need but to be more accurate)
        image = process_image(image)
        images = np.array([image])
        images = images.transpose((0, 3, 1, 2))
        images = torch.from_numpy(images).float()
        images = images.to(self.device)

        outputs = self.model(images)
        outputs = outputs.permute(1, 0, 2)
        output = outputs[0]

        out_best = list(torch.argmax(output, -1))  # [2:]
        out_best = [k for k, g in itertools.groupby(out_best)]
        pred_text = self.voc.get_label_from_indices(out_best)

        return pred_text, 1


if __name__ == '__main__':
    image = np.zeros([45, 100, 3])
    model = LionelOCR('saved/check', 'data/vocab.json')
    print(model.process(image))