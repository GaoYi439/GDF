import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def predict(Outputs, threshold):
    sig = nn.Sigmoid()
    pre_label = sig(Outputs)
    min_pre_label, _ = torch.min(pre_label.data, 1)
    max_pre_label, _ = torch.max(pre_label, 1)
    min_pre_label = min_pre_label.view(-1, 1)
    max_pre_label = max_pre_label.view(-1, 1)
    for i in range(len(pre_label)):
        if max_pre_label[i] - min_pre_label[i] != 0:
            pre_label[i, :] = (pre_label[i, :] - min_pre_label[i]) / (max_pre_label[i] - min_pre_label[i])
    pre = pre_label

    pre[pre > threshold] = 1
    pre[pre <= threshold] = 0
    return pre

def loss_unbiase(outputs, com_labels):
    '''
    for a drive loss based the bce loss, where l means -log or others loss function
    :param outputs: the presicted value with n*K size
    :param com_labels: the complementary label matrix, which is a one-hot vector for per instance
    :return: the loss value
    '''
    n, K = com_labels.size()[0], com_labels.size()[1]
    sig = nn.Sigmoid()
    sig_outputs = sig(outputs)
    pos_outputs = 1 - com_labels
    neg_outputs = com_labels

    part_1 = -torch.sum(pos_outputs * torch.log(sig_outputs + 1e-12), dim=1).mean()
    part_2 = -torch.sum(pos_outputs * torch.log(1.0 - sig_outputs + 1e-12), dim=1).mean()
    part_3 = -torch.sum(neg_outputs * torch.log(1.0 - sig_outputs + 1e-12), dim=1).mean()
    ave_loss = 2**(K-2)/(2**(K-1)-1) * part_1 + (2**(K-2) - 1)/(2**(K-1)-1) * part_2 + part_3
    return ave_loss

def GDF(outputs, com_labels):
    '''
    for a drive loss based the bce loss, where l means -log or others loss function.
    The loss is an upper-bound of loss_unbiase_1.
    :param outputs: the presicted value with n*K size
    :param com_labels: the complementary label matrix, which is a one-hot vector for per instance
    :return: the loss value
    '''
    sig = nn.Sigmoid()
    sig_outputs = sig(outputs)
    pos_outputs = 1 - com_labels
    neg_outputs = com_labels

    part_1 = -torch.sum(torch.log(sig_outputs + 1e-12) * pos_outputs, dim=1).mean()
    part_3 = -torch.sum(torch.log(1.0 - sig_outputs + 1e-12) * neg_outputs, dim=1).mean()
    ave_loss = part_1 + part_3
    return ave_loss
