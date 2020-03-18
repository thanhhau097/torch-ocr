import difflib
import itertools

import numpy as np
import torch
from ..data_loader.vocab import Vocab


def accuracy_attention(outputs, targets, voc: Vocab):
    outputs = _get_target_from_output(outputs)
    outputs = outputs.transpose(1, 0)
    targets = targets.transpose(1, 0)

    total_acc_by_char = 0
    total_acc_by_field = 0
    for output, target in zip(outputs, targets):
        pred_text = voc.get_label_from_indices(output)
        target_text = voc.get_label_from_indices(target)

        acc_by_char = calculate_ac(pred_text, target_text)
        total_acc_by_char += acc_by_char

        if pred_text == target_text:
            total_acc_by_field += 1
        # print("Predict:", pred_text)
        # print("Target:", target_text)

    return np.array([total_acc_by_char / targets.size()[0], total_acc_by_field / targets.size()[0]])


def accuracy_ctc(outputs, targets, voc: Vocab):
    outputs = outputs.permute(1, 0, 2)
    targets = targets.transpose(1, 0)

    total_acc_by_char = 0
    total_acc_by_field = 0

    for output, target in zip(outputs, targets):
        out_best = list(torch.argmax(output, -1))  # [2:]
        out_best = [k for k, g in itertools.groupby(out_best)]
        pred_text = voc.get_label_from_indices(out_best)
        target_text = voc.get_label_from_indices(target)

        # print('predict:', pred_text, 'target:', target_text)

        acc_by_char = calculate_ac(pred_text, target_text)
        total_acc_by_char += acc_by_char

        if pred_text == target_text:
            total_acc_by_field += 1

    return np.array([total_acc_by_char / targets.size()[0], total_acc_by_field / targets.size()[0]])


def calculate_ac(str1, str2):
    """Calculate accuracy by char of 2 string"""

    total_letters = len(str1)
    ocr_letters = len(str2)
    if total_letters == 0 and ocr_letters == 0:
        acc_by_char = 1.0
        return acc_by_char
    diff = difflib.SequenceMatcher(None, str1, str2)
    correct_letters = 0
    for block in diff.get_matching_blocks():
        correct_letters = correct_letters + block[2]
    if ocr_letters == 0:
        acc_by_char = 0
    elif correct_letters == 0:
        acc_by_char = 0
    else:
        acc_1 = correct_letters / total_letters
        acc_2 = correct_letters / ocr_letters
        acc_by_char = 2 * (acc_1 * acc_2) / (acc_1 + acc_2)

    return float(acc_by_char)


def _get_target_from_output(output):
    return torch.argmax(output, dim=-1)


if __name__ == '__main__':
    from ocr.data_loader.data_loaders import OCRDataLoader
    from ocr.data_loader.collate import collate_wrapper
    from ocr.model import AttentionModel

    dataloader = OCRDataLoader('../data', 'train.json', 4, collate_fn=collate_wrapper)
    item = next(iter(dataloader))
    print(item[1].size())
    print('Input size:', item[0].size())

    device = torch.device("cpu")
    model = AttentionModel(num_chars=65)
    x = model(item[0], item[1], item[3], device)
    print("After Decoder", x.size())

    predict = _get_target_from_output(x)
    print(predict.size())
