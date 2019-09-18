import torch.nn.functional as F
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)


def nll_mask_loss(outputs, targets, masks):
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")

    loss = 0
    # total_loss = total_loss.to(device)
    print_losses = []
    n_totals = 0

    for output, target, mask in zip(outputs, targets, masks):
        mask_loss, nTotal = sub_nll_mask_loss(output, target, mask, device)
        # print('loss:', loss, 'loss.item():', loss.item())
        loss += mask_loss
        print_losses.append(mask_loss.item() * nTotal)
        n_totals += nTotal
        # print('total_loss:', total_loss)
        # print(total_loss, print_losses)

    # print("LOSS:", loss, "PRINT_LOSSES:", sum(print_losses)/n_totals)
    return loss, sum(print_losses)/n_totals


def sub_nll_mask_loss(output, target, mask, device):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(output, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()
