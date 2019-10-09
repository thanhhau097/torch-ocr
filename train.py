import argparse
import collections

import ocr.data_loader.data_loaders as module_data
import ocr.model.loss as module_loss
import torch
from ocr.model import ocr_model as module_arch

import ocr.model.metric as module_metric
from ocr.parse_config import ConfigParser
from ocr.trainer import Trainer


def main(config):
    logger = config.get_logger('train')
    model_type = config['type']

    # setup data_loader instances
    data_loader = config.initialize('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # get vocab -> pass num chars to model
    voc = data_loader.get_vocab()
    kwarg = {"num_chars": voc.num_chars}

    # build model architecture, then print to console
    model = config.initialize('arch', module_arch, **kwarg)
    logger.info(model)

    # get function handles of loss and metrics
    loss = getattr(module_loss, config['loss'][model_type])
    metrics = [getattr(module_metric, met) for met in config['metrics'][model_type]]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, loss, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
    ]
    config = ConfigParser(args, options)
    main(config)
