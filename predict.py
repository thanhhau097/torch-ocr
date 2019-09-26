import torch
from model.ctc_model import CTCModel
from data_loader.vocab import Vocab
import numpy as np
from data_loader.collate import process_image
import itertools

# TODO: currently only work with CTC model
def process(image):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    voc = Vocab()
    voc.build_vocab_from_char_dict_file('data/vocab.json')

    model = CTCModel(voc.num_chars)

    model = model.to(device)

    # # LOAD MODEL
    # weights_path = 'saved/models/OCR_test/0920_132220/checkpoint-epoch42.pth'
    # checkpoint = torch.load(weights_path)
    # state_dict = checkpoint['state_dict']
    # model.load_state_dict(state_dict)

    model.eval()

    # preprocess image, TODO: height = 64 (no need but to be more accurate)
    image = process_image(image)
    images = np.array([image])
    images = images.transpose((0, 3, 1, 2))
    images = torch.from_numpy(images).float()
    images = images.to(device)
    outputs = model(images)
    outputs = outputs.permute(1, 0, 2)
    output = outputs[0]
    out_best = list(torch.argmax(output, -1))  # [2:]
    out_best = [k for k, g in itertools.groupby(out_best)]
    pred_text = voc.get_label_from_indices(out_best)
    print(pred_text)

if __name__ == '__main__':
    image = np.zeros([45, 100, 3])
    process(image)