import argparse

import numpy as np
import torch
from tqdm import tqdm

import ocr.data_loader.data_loaders as module_data
import ocr.model.loss as module_loss
import ocr.model.metric as module_metric
import ocr.model.ocr_model as module_arch
from ocr.parse_config import ConfigParser


def main(config):
    # TODO: not found char from train_dataset -> error when validate
    logger = config.get_logger('test')

    # setup data_loader instances
    json_path = 'daiichi4.json'
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        json_path,
        training=False,
        batch_size=1,
        shuffle=False,
        validation_split=0.0,
        num_workers=2
    )

    # get vocab -> pass num chars to model
    voc = data_loader.get_vocab()
    kwarg = {"num_chars": voc.num_chars}

    # build model architecture
    model = config.initialize('arch', module_arch, **kwarg)
    logger.info(model)

    #
    model_type = config['type']

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'][model_type])
    metric_fns = [getattr(module_metric, met) for met in config['metrics'][model_type]]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    if torch.cuda.is_available():
        checkpoint = torch.load(config.resume)
    else:
        checkpoint = torch.load(config.resume, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = np.zeros(2)
    # total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (images, labels, mask, max_label_length) in enumerate(tqdm(data_loader)):
            images, labels, mask = images.to(device), labels.to(device), mask.to(device)

            output = model(images, labels, max_label_length, device, training=False)
            # loss, print_losses = self.loss(output, labels, mask)  # Attention:
            # lengths = torch.sum(mask, dim=0).to(device)
            loss, print_loss = loss_fn(output, labels, mask)

            # batch_size = data.shape[0]
            total_loss += print_loss  # loss.item() * batch_size
            # for i, metric in enumerate(metric_fns):
            #     total_metrics[i] += metric(output, target) * batch_size
            total_metrics += metric_fns[0](output, labels, voc)

    # n_samples = len(data_loader.sampler)
    n_batches = len(data_loader)
    log = {'loss': total_loss / n_batches}
    # log.update({
    #     met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    # })
    log.update({
        metric_fns[0].__name__: total_metrics / n_batches
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume', default='saved/model_best_real_data_2.pth', type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-c', '--config', default='ocr/config.json', type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser(args)
    main(config)
