import json
import os

import cv2
from torch.utils.data import Dataset

from ..data_loader.vocab import Vocab


# TODO: add training=True/False, when testing, we need to load vocab, not build vocab
class OCRDataset(Dataset):
    """Read dataset for OCR"""
    def __init__(self, data_dir, json_path, transform=None, training=True, channels=3):
        self.data_dir = data_dir
        self.json_path = json_path
        self.transform = transform
        self.channels = channels

        self.image_paths, self.labels = self.__get_image_paths_and_labels(self.get_data_path(json_path))
        if training:
            self.voc = self.build_vocab(self.labels)
            self.voc.save_vocab_dict()
        else:
            self.voc = Vocab()
            self.voc.build_vocab_from_char_dict_file(self.get_data_path('vocab.json'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.get_data_path(self.image_paths[idx])
        image = self.read_image(img_path)
        label = self.labels[idx]
        label = self.voc.get_indices_from_label(label)
        sample = {"image": image, "label": label}
        return sample

    def read_image(self, path):
        if self.channels == 1:
            img = cv2.imread(path, 0)
        elif self.channels == 3:
            img = cv2.imread(path)
        else:
            raise ValueError("Number of channels must be 1 or 3")

        return img

    def get_data_path(self, path):
        return os.path.join(self.data_dir, path)

    def __get_image_paths_and_labels(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        image_paths = list(data.keys())
        labels = list(data.values())
        return image_paths, labels

    def build_vocab(self, labels):
        voc = Vocab()
        for label in labels:
            voc.add_label(label)

        return voc

    def get_vocab(self):
        return self.voc

if __name__ == '__main__':
    dataset = OCRDataset('../data', 'train.json')
    for i, item in enumerate(dataset):
        print(item['label'])
        if i == 10:
            break
