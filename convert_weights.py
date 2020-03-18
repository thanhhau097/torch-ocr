"""Convert weight to use in Server"""

import os

import torch


def main():
    folder = '/Users/macos/Desktop/Jobs/Self/ocr/ocr_product/model'
    weights_path = os.path.join(folder, 'weights/model_best.pth')
    checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    torch.save({'state_dict': state_dict}, os.path.join(folder, 'weights/serving_model.pth'))
    print('convert success')


if __name__ == '__main__':
    main()
