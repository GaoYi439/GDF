import argparse
import torch.nn as nn
import torch
import numpy as np
from torch.backends import cudnn
import random

from utils.metrics import OneError, Coverage, HammingLoss, RankingLoss, AveragePrecision
from utils.models import linear
from utils.utils_algo import adjust_learning_rate, predict, loss_unbiase, GDF
from utils.utils_data import choose

parser = argparse.ArgumentParser(description='PyTorch implementation of IJCAI 2023 paper GDF')
parser.add_argument('--dataset', default='scene', type=str, choices=['scene', 'yeast'], help='dataset name')
parser.add_argument('--num-class', default=6, type=int, help='number of classes')
parser.add_argument('--input-dim', default=294, type=int, help='number of features')
parser.add_argument('--fold', default=9, type=int, help='fold-th fold of 10-cross fold')
parser.add_argument('--model', default="linear", type=str, choices=['linear'])
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--wd', default=1e-3, type=float, help='weight decay')
parser.add_argument('--batch_size', default=256, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--schedule', default=[100, 150], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--lo', default="GDF", type=str, help='learning loss function)')
parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
parser.add_argument('--the', default=0.8, type=float, help='prediction threshold. ')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    print(args)

    cudnn.benchmark = True

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # make data
    train_loader, test_loader, args.num_class, args.input_dim = choose(args)

    # choose model
    if args.model == "linear":
        model = linear(input_dim=args.input_dim, output_dim=args.num_class)

    model = model.to(device)

    # set optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9,
                                weight_decay=args.wd)

    print("start training")

    best_av = 0
    save_table = np.zeros(shape=(args.epochs, 7))

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        train_loss = train(train_loader, model, optimizer, args)
        t_hamm, t_one_error, t_converage, t_rank, t_av_pre = validate(test_loader, model, args)
        print("Epoch:{ep}, Tr_loss:{tr}, T_hamm:{T_hamm}, T_one_error:{T_one_error}, T_con:{T_con}, "
              "T_rank:{T_rank}, T_av:{T_av}".format(ep=epoch, tr=train_loss, T_hamm=t_hamm, T_one_error=t_one_error,
                                                    T_con=t_converage, T_rank=t_rank, T_av=t_av_pre))
        save_table[epoch, :] = epoch + 1, train_loss, t_hamm, t_one_error, t_converage, t_rank, t_av_pre

        np.savetxt("result/{ds}_{md}_{M}_lr{lr}_wd{wd}_fold{fd}.csv".format(ds=args.dataset, md=args.lo,
                                                                                  M=args.model, lr=args.lr, wd=args.wd,
                                                                                  fd=args.fold), save_table,
                   delimiter=',', fmt='%1.4f')
        # save model
        if t_av_pre > best_av:
            best_av = t_av_pre
            torch.save(model.state_dict(), "experiment/{ds}_{md}_{M}_lr{lr}_wd{wd}_fold{fd}_best_model.tar".format(
                ds=args.dataset, md=args.lo, M=args.model, lr=args.lr, wd=args.wd, fd=args.fold))



def train(train_loader, model, optimizer, args):
    model.train()
    train_loss = 0
    for i, (images, _, com_labels, index) in enumerate(train_loader):
        images, com_labels = images.to(device), com_labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        if args.lo == "unbiase":
            loss = loss_unbiase(outputs, com_labels)
        elif args.lo == "GDF":
            loss = GDF(outputs, com_labels)
        loss.backward()
        optimizer.step()

        train_loss = train_loss + loss.item()

    return train_loss / len(train_loader)


# test the results, all data in the test data is placed in a batch
def validate(test_loader, model, args):
    with torch.no_grad():
        model.eval()
        sig = nn.Sigmoid()
        for data, targets, _, _ in test_loader:
            images, targets = data.to(device), targets.to(device)
            output = model(images)
            pre_output = sig(output)
            pre_label = predict(output, args.the)

    t_one_error = OneError(pre_output, targets)
    t_converage = Coverage(pre_output, targets)
    t_hamm = HammingLoss(pre_label, targets)
    t_rank = RankingLoss(pre_output, targets)
    t_av_pre = AveragePrecision(pre_output, targets)

    return t_hamm, t_one_error, t_converage, t_rank, t_av_pre


if __name__ == '__main__':
    lr_1e_1 = ["yeast"]
    lr_1e_2 = ["scene"]

    args = parser.parse_args()
    if args.dataset in lr_1e_1:
        args.lr = 0.1
    elif args.dataset in lr_1e_2:
        args.lr = 0.01

    for fd in range(10):
        args.fold = fd
        print(
            "Data:{ds}, model:{model}, lr:{lr}, wd:{wd}, fold:{fd}, loss:{lo}".format(ds=args.dataset, model=args.model,
                                                                                      lr=args.lr, wd=args.wd,
                                                                                      fd=args.fold, lo=args.lo))
        main()
