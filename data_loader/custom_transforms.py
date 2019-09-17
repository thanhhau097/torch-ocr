import torch
from torchvision.transforms.functional import resize


class ToTensor(object):
    """Convert elements in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        height (tuple or int): Desired height.
    """

    def __init__(self, height):
        assert isinstance(height, int)
        self.height = height

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.height, w * self.height / h
        new_h, new_w = int(new_h), int(new_w)

        img = resize(image, (new_h, new_w))
        return {'image': img, 'label': label}
