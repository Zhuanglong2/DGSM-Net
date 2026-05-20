#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from dataset import dataset_dict
from engine import test_one_epoch_seq, train_one_epoch_seq
from models import PGSMNet
from utils import parse_args_from_yaml


class IOStream:
    def __init__(self, path):
        self.f = open(path, "a")

    def cprint(self, text):
        print(text)
        self.f.write(text + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def prepare_dirs(args):
    os.makedirs(f"checkpoints/{args.exp_name}/models", exist_ok=True)
    os.makedirs(f"checkpoints/{args.exp_name}/loss_train", exist_ok=True)


def init_model(args):
    net = PGSMNet(args).cuda()
    if args.eval or args.load_checkpoint:
        model_path = args.model_path or f"checkpoints/{args.exp_name}/models/model.best.t7"
        print(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Cannot find model checkpoint: {model_path}")
        net.load_state_dict(torch.load(model_path), strict=False)
        print("Successfully loaded model parameters.")
    return net


def train(args, net, train_loader, test_loader, textio, tb_logger):
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, args.decay_epochs, gamma=args.decay_rate)
    best_val_res = np.inf

    for epoch in range(args.epochs):
        textio.cprint("====epoch: %d, learning rate: %f====" % (epoch, optimizer.param_groups[0]["lr"]))
        textio.cprint("==starting training on the training set==")
        total_loss, loss_items = train_one_epoch_seq(args, net, train_loader, optimizer)
        textio.cprint("mean train loss: %f" % total_loss)
        tb_logger.add_scalar("total_loss", total_loss, epoch)
        tb_logger.add_scalar("odm_loss", loss_items["odmLoss"], epoch)
        tb_logger.add_scalar("tLoss", loss_items["tLoss"], epoch)
        tb_logger.add_scalar("qLoss", loss_items["qLoss"], epoch)

        textio.cprint("==starting evaluation on the validation/test set==")
        _, pose_metric_kitti = test_one_epoch_seq(args, net, test_loader, textio)
        eval_trmse = pose_metric_kitti["TRMSE"]
        eval_rrmse = pose_metric_kitti["RRMSE"]
        textio.cprint("TRMSE score: %f" % eval_trmse)
        textio.cprint("RRMSE score: %f" % eval_rrmse)
        tb_logger.add_scalar("TRMSE", eval_trmse, epoch)
        tb_logger.add_scalar("RRMSE", eval_rrmse, epoch)

        score = (eval_trmse + eval_rrmse * 100) / 2
        if best_val_res >= score:
            best_val_res = score
            torch.save(net.state_dict(), f"checkpoints/{args.exp_name}/models/model.best.t7")
            textio.cprint("best val TRMSE till now: %f" % eval_trmse)
            textio.cprint("best val RRMSE till now: %f" % eval_rrmse)
        torch.save(net.state_dict(), f"checkpoints/{args.exp_name}/models/model{epoch}.t7")
        scheduler.step()


def run(cli_args):
    args = parse_args_from_yaml(cli_args.config)
    if cli_args.eval:
        args.eval = True

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    torch.cuda.empty_cache()
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)

    prepare_dirs(args)
    textio = IOStream(f"checkpoints/{args.exp_name}/run.log")
    textio.cprint(str(args))
    tb_logger = SummaryWriter(f"checkpoints/{args.exp_name}/train_tb")

    if args.eval:
        test_set = dataset_dict[args.dataset](args=args, root=args.dataset_path, partition=args.eval_split, textio=textio)
        test_loader = DataLoader(test_set, num_workers=args.num_workers, batch_size=1, shuffle=False, drop_last=False)
        args.camera_projection_matrix = test_set.camera_projection_matrix
        args.t_camera_radar = test_set.t_camera_radar
        args.radar_res = test_set.res
        args.clips_info = test_set.clips_info
        net = init_model(args)
        pose_metric, pose_metric_kitti = test_one_epoch_seq(args, net, test_loader, textio)
        for key, value in pose_metric.items():
            textio.cprint("###The mean {}: {}###".format(key, value))
        textio.cprint("TRMSE score: %f" % pose_metric_kitti["TRMSE"])
        textio.cprint("RRMSE score: %f" % pose_metric_kitti["RRMSE"])
    else:
        train_set = dataset_dict[args.dataset](args=args, root=args.dataset_path, partition=args.train_set, textio=textio)
        args.eval = True
        test_set = dataset_dict[args.dataset](args=args, root=args.dataset_path, partition=args.eval_split, textio=textio)
        args.eval = False
        train_loader = DataLoader(train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_set, num_workers=args.num_workers, batch_size=1, shuffle=False, drop_last=False)
        args.camera_projection_matrix = train_set.camera_projection_matrix
        args.t_camera_radar = train_set.t_camera_radar
        args.radar_res = train_set.res
        args.clips_info = test_set.clips_info
        net = init_model(args)
        train(args, net, train_loader, test_loader, textio, tb_logger)

    textio.cprint("Max memory allocation: {}MB".format(torch.cuda.max_memory_allocated(device=0) / 1e6))
    textio.cprint("FINISH")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PGSM-Net training/evaluation")
    parser.add_argument("--config", type=str, default="configs.yaml")
    parser.add_argument("--eval", action="store_true")
    run(parser.parse_args())
