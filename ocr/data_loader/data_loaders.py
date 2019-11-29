from ..base import BaseDataLoader
from ..data_loader.dataset import OCRDataset
from torchvision import transforms

from ..data_loader.collate import collate_wrapper


class OCRDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, json_path, batch_size,
                 collate_fn=collate_wrapper,
                 training=True,
                 shuffle=True,
                 validation_split=0.0, num_workers=1):
        # self.height = 32
        trsfm = transforms.Compose([
            # ToTensor(),
            # Rescale(self.height),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.json_path = json_path
        self.dataset = OCRDataset(data_dir, json_path, trsfm, training=training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)

    def get_vocab(self):
        return self.dataset.get_vocab()


if __name__ == '__main__':
    dataloader = OCRDataLoader('../data', 'train.json', 4, collate_fn=collate_wrapper)
    for item in dataloader:
        print(item)
        break
