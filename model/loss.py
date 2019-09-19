import torch.nn.functional as F
import torch
from torch.nn import CTCLoss
from data_loader.vocab import PAD_token, EOS_token


def nll_loss(output, target):
    return F.nll_loss(output, target)


def nll_mask_loss(outputs, targets, masks):
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")

    loss = 0
    print_losses = []
    n_totals = 0

    for output, target, mask in zip(outputs, targets, masks):
        mask_loss, nTotal = sub_nll_mask_loss(output, target, mask, device)
        loss += mask_loss
        print_losses.append(mask_loss.item() * nTotal)
        n_totals += nTotal

    return loss, sum(print_losses)/n_totals


def sub_nll_mask_loss(output, target, mask, device):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(output, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


def ctc_loss(outputs, targets, mask):
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    target_lengths = torch.sum(mask, dim=0).to(device)
    # We need to change targets, PAD_token = 0 = blank
    # EOS token -> PAD_token
    targets[targets == EOS_token] = PAD_token
    outputs = outputs.log_softmax(2)
    input_lengths = outputs.size()[0] * torch.ones(outputs.size()[1], dtype=torch.int)
    loss_fn = CTCLoss(blank=PAD_token, zero_infinity=True)
    targets = targets.transpose(1, 0)
    # target_lengths have EOS token, we need minus one
    target_lengths = target_lengths - 1
    targets = targets[:, :-1]
    # print(input_lengths, target_lengths)
    # TODO: bug when target_length > input_length, we can increase size or use zero infinity
    loss = loss_fn(outputs, targets, input_lengths, target_lengths)
    return loss, loss.item()
