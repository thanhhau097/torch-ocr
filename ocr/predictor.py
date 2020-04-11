import itertools

import numpy as np
import torch
from .data_loader.vocab import Vocab
from .model.ctc_model import CTCModel

from .data_loader.collate import collate_wrapper


class OCRModule:
    def __init__(self, weights_path, vocab_path):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.voc = Vocab()
        self.voc.build_vocab_from_char_dict_file(vocab_path)
        self.model = CTCModel(self.voc.num_chars)
        self.model = self.model.to(self.device)

        # LOAD MODEL
        print("LOADING PRETRAINED WEIGHTS ...")
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
        state_dict = checkpoint['state_dict']
        self.model.load_state_dict(state_dict)
        print("LOAD WEIGHTS SUCCESSFULLY")

    def process(self, image):
        self.model.eval()
        # preprocess image, TODO: height = 64 (no need but to be more accurate)
        batch = [{'image': image, 'label': [1]}]
        images = collate_wrapper(batch)[0]

        # image = process_image(image)
        # images = np.array([image])
        # images = images.transpose((0, 3, 1, 2))
        # images = torch.from_numpy(images).float()
        images = images.to(self.device)

        outputs = self.model(images)
        outputs = outputs.permute(1, 0, 2)
        output = outputs[0]

        out_best = list(torch.argmax(output, -1))  # [2:]
        out_best = [k for k, g in itertools.groupby(out_best)]
        pred_text = self.voc.get_label_from_indices(out_best)

        return pred_text, 1


def use_rules(text):
    char_dict = {
        '0': '0',
        'O': '0',
        'つ': 'っ'
    }
    ignored_characters = [',', '。', '.', '、']
    for char in ignored_characters:
        text = text.replace(char, '')

    for k, v in char_dict.items():
        text = text.replace(k, v)

    return ''.join(text.split(' '))


if __name__ == '__main__':
    import cv2
    import json
    import os

    path = 'saved/model_best_real_data_2.pth'
    model = OCRModule(path, 'data/vocab.json')

    with open('data/daiichi4/val.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_true = 0
    for i, (name, label) in enumerate(data.items()):
        path = os.path.join('data/daiichi4/', name)  # 'data/sample/56/5/264_ENGROSSING_25813.jpg'
        image = cv2.imread(path)
        # padding
        padding = 0
        new_image = np.zeros([image.shape[0] + padding * 2, image.shape[1] + padding * 2, image.shape[2]])
        new_image[padding:image.shape[0] + padding, padding:image.shape[1] + padding, :] = image
        predicted = use_rules(label)
        ground_truth = use_rules(model.process(new_image)[0])
        total_true += int(predicted == ground_truth)
        # print(use_rules(label))
        # print(use_rules(model.process(new_image)[0]))
        if i % 100 == 0:
            print(total_true, '/', i+1, '=', total_true/(i + 1))