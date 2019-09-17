PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


class Vocab(object):
    def __init__(self):
        # Default word tokens
        self.char2index = {}
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

    def get_indices_from_label(self, label):
        indices = []
        for char in label:
            indices.append(self.char2index[char])

        indices.append(EOS_token)
        return indices
