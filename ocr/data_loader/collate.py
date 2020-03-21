import itertools

import cv2
import numpy as np
import torch

from ..data_loader.vocab import PAD_token


class OCRCollate:
    def __init__(self):
        pass

    def __call__(self, batch):
        """
        Pad images, pad output, mask for output, max_output_length

        :param batch: batch of data samples
        :return:
        """
        for sample in batch:
            print(sample['label'])


def collate_wrapper(batch):
    """
    Labels are already numbers

    :param batch:
    :return:
    """
    images = []
    labels = []
    # TODO: can change height in config
    height = 64
    max_width = 0
    max_label_length = 0

    for sample in batch:
        image = sample['image']
        try:
            image = process_image(image, height=height, channels=image.shape[2])
        except:
            continue

        if image.shape[1] > max_width:
            max_width = image.shape[1]

        label = sample['label']

        if len(label) > max_label_length:
            max_label_length = len(label)

        images.append(image)
        labels.append(label)

    # PAD IMAGES: convert to tensor with size b x c x h x w (from b x h x w x c)
    channels = images[0].shape[2]
    images = process_batch_images(images, height=height, max_width=max_width, channels=channels)
    images = images.transpose((0, 3, 1, 2))
    images = torch.from_numpy(images).float()

    # LABELS
    pad_list = zero_padding(labels)
    mask = binary_matrix(pad_list)
    mask = torch.ByteTensor(mask)
    labels = torch.LongTensor(pad_list)
    return images, labels, mask, max_label_length


def process_image(image, height=64, channels=3):
    """Converts to self.channels, self.max_height
    # convert channels
    # resize max_height = 64
    """
    shape = image.shape
    # if shape[0] > 64 or shape[0] < 32:  # height
    try:
        image = cv2.resize(image, (int(height/shape[0] * shape[1]), height))
    except:
        return np.zeros([1, 1, channels])
    return image / 255.0


def process_batch_images(images, height, max_width, channels=3):
    """
    Convert a list of images to a tensor (with padding)

    :param images: list of numpy array images
    :param height: desired height
    :param max_width: max width of all images
    :param channels: number of image channels
    :return: a tensor representing images
    """
    output = np.ones([len(images), height, max_width, channels])
    for i, image in enumerate(images):
        final_img = image
        shape = image.shape
        output[i, :shape[0], :shape[1], :] = final_img

    return output


def zero_padding(l, fillvalue=PAD_token):
    """
    Pad value PAD token to l
    :param l: list of sequences need padding
    :param fillvalue: padded value
    :return:
    """
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binary_matrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == value:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


if __name__ == '__main__':
    pass
