import time

import numpy as np
import torch
from torchvision.utils import make_grid

from ..base import BaseTrainer
from ..utils import inf_loop


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.voc = self.data_loader.get_vocab()
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, output, target):
        # acc_metrics = np.zeros(len(self.metrics))
        acc_metrics = np.zeros(2)
        acc_metrics += self.metrics[0](output, target, self.voc)
        # for i, metric in enumerate(self.metrics):
        #     acc_metrics[i] += metric(output, target, self.voc)
        #     self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        # total_metrics = np.zeros(len(self.metrics))
        total_metrics = np.zeros(2)

        # end = time.time()
        for batch_idx, (images, labels, mask, max_label_length) in enumerate(self.data_loader):
            # start = time.time()
            # print("Time load batch:", start - end)
            images, labels, mask = images.to(self.device), labels.to(self.device), mask.to(self.device)
            # print("After change device:", images.get_device(), labels.get_device(), mask.get_device())

            self.optimizer.zero_grad()
            output = self.model(images, labels, max_label_length, self.device)

            loss, print_loss = self.loss(output, labels, mask)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.writer.add_scalar('loss', print_loss)
            total_loss += print_loss  # loss.item()
            total_metrics += self._eval_metrics(output, labels)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    print_loss  # loss.item()
                ))
                self.writer.add_image('input', make_grid(images.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break

            # end = time.time()
            # print("Time training batch:", end - start)

        log = {
            'loss': total_loss / self.len_epoch,
            'metrics': (total_metrics / self.len_epoch).tolist()
        }

        print(log)

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        # when evaluating, we don't use teacher forcing
        self.model.eval()
        total_val_loss = 0
        # total_val_metrics = np.zeros(len(self.metrics))
        total_val_metrics = np.zeros(2)
        with torch.no_grad():
            # print("Length of validation:", len(self.valid_data_loader))
            for batch_idx, (images, labels, mask, max_label_length) in enumerate(self.valid_data_loader):
                images, labels, mask = images.to(self.device), labels.to(self.device), mask.to(self.device)
                images, labels, mask = images.to(self.device), labels.to(self.device), mask.to(self.device)

                output = self.model(images, labels, max_label_length, self.device, training=False)
                _, print_loss = self.loss(output, labels, mask)  # Attention:
                # loss = self.loss(output, labels, mask)
                # print_loss = loss.item()

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', print_loss)
                total_val_loss += print_loss
                total_val_metrics += self._eval_metrics(output, labels)
                self.writer.add_image('input', make_grid(images, nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return_value = {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
        print(return_value)

        return return_value

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
