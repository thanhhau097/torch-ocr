import json


PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


class Vocab(object):
    def __init__(self):
        # Default word tokens
        self.char2index = {"PAD": PAD_token, "SOS": SOS_token, "EOS": EOS_token}
        self.index2char = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_chars = 3

    def add_label(self, label):
        for char in label:
            self.add_char(char)

    def add_char(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.num_chars
            self.index2char[self.num_chars] = char
            self.num_chars += 1

    def build_vocab_from_char_dict_file(self, json_path):
        with open(json_path, 'r') as f:
            vocab_dict = json.load(f)

        for key, value in vocab_dict.items():
            self.index2char[int(key)] = value
            self.char2index[value] = int(key)
            self.num_chars += 1

    def save_vocab_dict(self):
        save_dict = {}
        for key, value in self.index2char.items():
            if key > 2:
                save_dict[key] = value

        with open('data/vocab.json', 'w') as f:
            json.dump(save_dict, f)

    def get_indices_from_label(self, label):
        indices = []
        for char in label:
            if self.char2index.get(char) is not None:
                indices.append(self.char2index[char])

        indices.append(EOS_token)
        return indices

    def get_label_from_indices(self, indices):
        label = ""
        for index in indices:
            if index == EOS_token:
                break
            elif index == PAD_token:
                continue
            else:
                label += self.index2char[index.item()]

        return label
