from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.dataset import OCRDataset


class OCRDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, json_path, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
            # TODO: resize image to same height
        ])
        self.data_dir = data_dir
        self.json_path = json_path
        self.dataset = OCRDataset(data_dir, json_path, trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


if __name__ == '__main__':
    dataloader = OCRDataLoader()
