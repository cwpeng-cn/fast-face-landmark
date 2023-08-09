# -*- encoding: utf-8 -*-
'''
Description      : train
Time             :2023/05/24 13:21:21
Author           :cwpeng
email            :cw.peng@foxmail.com
'''
import os
import torch
import logging
import argparse
import numpy as np

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.utils.data import random_split
from torch.cuda.amp import GradScaler, autocast

from dataset.aflw_dataset import AFLWDataset
from model.mobileone import mobileone
from loss.weight_loss import WeightLoss
from utils.avg_meter import AverageMeter


device = "cuda"
scaler = GradScaler()
torch.manual_seed(0)


def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info('Save checkpoint to {0:}'.format(filename))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def train(train_loader, net, criterion, optimizer):
    losses = AverageMeter()
    net = net.to(device)

    weighted_loss = None
   
    for item in train_loader:

        img = item["img"].to(device)
        landmark_gt = item["keypoints"].to(device)
        visable = item["visable"].to(device)
        actually_visable = item['actually_visable'].to(device)
        weight = item['weight'].to(device)

        optimizer.zero_grad()

        with autocast():
            pre_landmarks, pre_visable = net(img)
            weighted_loss = criterion(landmark_gt, pre_landmarks, visable,
                                      pre_visable, actually_visable, weight)
            print(weighted_loss.item())

        scaler.scale(weighted_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.update(weighted_loss.item())  

    return weighted_loss


def validate(val_dataloader, net, criterion):
    net.eval().to(device)
    losses = []
    with torch.no_grad():
        for item in val_dataloader:
            img = item["img"].to(device)
            landmark_gt = item["keypoints"].to(device)
            visable = item["visable"].to(device)
            actually_visable = item['actually_visable'].to(device)
            weight = item['weight'].to(device)

            pre_landmark, pre_visable = net(img)
            weighted_loss = criterion(landmark_gt, pre_landmark, visable,
                                      pre_visable, actually_visable, weight)
            losses.append(weighted_loss.cpu().numpy())
    print("===> Evaluate:")
    print('Eval set: Average loss: {:.4f} '.format(np.mean(losses)))
    return np.mean(losses)


def main(args):
    # Step 1: parse args config
    logging.basicConfig(
        format=
        '[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_file, mode='w'),
            logging.StreamHandler()
        ])
    print_args(args)

    # Step 2: model, criterion, optimizer, scheduler
    # net = PFLDInference().to(device)
    net = mobileone(num_classes=68 * 2, inference_mode=False, variant="s3")
    criterion = WeightLoss()
    optimizer = torch.optim.Adam([{
        'params': net.parameters()
    }],
                                 lr=args.base_lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=args.lr_patience, verbose=True)
    if args.resume:
        checkpoint = torch.load(args.resume)
        net.load_state_dict(checkpoint["net"])
        args.start_epoch = checkpoint["epoch"]

    # step 3: data
    # argumetion
    dataset = AFLWDataset()
    train_ds, val_ds = random_split(dataset,
                                    lengths=[
                                        int(len(dataset) * 0.95),
                                        len(dataset) -
                                        int(len(dataset) * 0.95)
                                    ])
    dataloader = DataLoader(train_ds,
                            batch_size=args.train_batchsize,
                            shuffle=True,
                            num_workers=args.workers,
                            drop_last=False)
    val_dataloader = DataLoader(val_ds,
                                batch_size=args.val_batchsize,
                                shuffle=False,
                                num_workers=args.workers)

    # step 4: run
    writer = SummaryWriter(args.tensorboard)
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        train_loss = train(dataloader, net, criterion, optimizer)
        filename = os.path.join(str(args.snapshot),
                                "checkpoint_epoch_" + str(epoch) + '.pth.tar')
        save_checkpoint({
            'epoch': epoch,
            'net': net.state_dict(),
        }, filename)

        val_loss = validate(val_dataloader, net, criterion)

        scheduler.step(val_loss)
        writer.add_scalars('data/loss', {
            'val loss': val_loss,
            'train loss': train_loss
        }, epoch)
    writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description='mobileone')
    # general
    parser.add_argument('-j', '--workers', default=8, type=int)
    parser.add_argument('--devices_id', default='0', type=str)  # TBD
    parser.add_argument('--test_initial', default='false',
                        type=str2bool)  # TBD

    # training
    ##  -- optimizer
    parser.add_argument('--base_lr', default=0.0001, type=int)
    parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float)

    # -- lr
    parser.add_argument("--lr_patience", default=40, type=int)

    # -- epoch
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--end_epoch', default=150, type=int)

    # -- snapshot、tensorboard log and checkpoint
    parser.add_argument('--snapshot',
                        default='./checkpoint/snapshot/',
                        type=str,
                        metavar='PATH')
    parser.add_argument('--log_file',
                        default="./checkpoint/train.logs",
                        type=str)
    parser.add_argument('--tensorboard',
                        default="./checkpoint/tensorboard",
                        type=str)
    parser.add_argument('--resume', default='', type=str, metavar='PATH')

    parser.add_argument('--train_batchsize', default=256, type=int)
    parser.add_argument('--val_batchsize', default=256, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
