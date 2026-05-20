import copy
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from KITTI_odometry_evaluation_tool.evaluation_delft import kittiOdomEval
from losses import RadarOdmLoss, loss_dict
from utils import eval_trans_RPE


def extract_data_info_clip(seq_data, idx):
    pc1, pc2, ft1, ft2, trans = seq_data
    pc1 = pc1[:, idx].cuda().transpose(2, 1).contiguous()
    pc2 = pc2[:, idx].cuda().transpose(2, 1).contiguous()
    ft1 = ft1[:, idx].cuda().transpose(2, 1).contiguous()
    ft2 = ft2[:, idx].cuda().transpose(2, 1).contiguous()
    trans = trans[:, idx].cuda().float()
    return pc1, pc2, ft1, ft2, trans


def extract_data_info(data):
    pc1, pc2, ft1, ft2, trans = data
    pc1 = pc1.cuda().transpose(2, 1).contiguous()
    pc2 = pc2.cuda().transpose(2, 1).contiguous()
    ft1 = ft1.cuda().transpose(2, 1).contiguous()
    ft2 = ft2.cuda().transpose(2, 1).contiguous()
    trans = trans.cuda().float()
    return pc1, pc2, ft1, ft2, trans


def train_one_epoch_seq(args, net, train_loader, optimizer):
    total_loss = 0.0
    num_examples = 0
    net.train()
    loss_items = copy.deepcopy(loss_dict["pgsm"])
    seq_len = train_loader.dataset.mini_clip_len
    loss_obj = RadarOdmLoss()

    for _, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        iter_loss = 0.0
        iter_items = copy.deepcopy(loss_dict["pgsm"])
        batch_size = data[0].shape[0]
        num_examples += batch_size
        gfeat = None

        for j in range(seq_len):
            pc1, pc2, ft1, ft2, gt_trans = extract_data_info_clip(data, j)
            if gfeat is not None:
                gfeat = gfeat.detach()
            odm_q, odm_t, w_x, w_q, a_x, a_q, _, gfeat = net(
                pc1, pc2, ft1, ft2, None, "train", gfeat
            )
            loss, _, _, items = loss_obj(gt_trans, odm_q, odm_t, w_x, w_q, a_x, a_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_loss += loss
            for key in iter_items:
                iter_items[key].append(items[key])

        iter_loss = iter_loss / seq_len
        for key in loss_items:
            loss_items[key].append(np.mean(np.asarray(iter_items[key])))
        total_loss += iter_loss.item() * batch_size

    total_loss = total_loss / max(num_examples, 1)
    for key in loss_items:
        loss_items[key] = np.mean(np.asarray(loss_items[key]))
    return total_loss, loss_items


def test_one_epoch_seq(args, net, test_loader, textio):
    net.eval()
    pose_metric = {"RTE": 0.0, "RAE": 0.0}
    gt_trans_all = torch.zeros((len(test_loader), 4, 4)).cuda()
    pre_trans_all = torch.zeros((len(test_loader), 4, 4)).cuda()
    infer_time_all = []

    with torch.no_grad():
        clips_info = test_loader.dataset.clips_info
        clip_starts = [clip["index"][0] for clip in clips_info]
        seq_len = test_loader.dataset.update_len
        num_clip = 0
        gfeat = None
        num_pcs = 0

        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            pc1, pc2, ft1, ft2, trans = extract_data_info(data)
            torch.cuda.synchronize()
            start_time = time.time()

            if i == clip_starts[num_clip] or i % seq_len == 0:
                gfeat = None
                if num_clip < len(clip_starts) - 1:
                    num_clip += 1

            _, _, _, _, _, _, pred_t, gfeat = net(pc1, pc2, ft1, ft2, None, "test", gfeat)

            torch.cuda.synchronize()
            infer_time_all.append(time.time() - start_time)

            gt_trans_all[num_pcs:num_pcs + 1] = trans
            pre_trans_all[num_pcs:num_pcs + 1] = pred_t
            pose_res = eval_trans_RPE(trans, pred_t)
            for key in pose_res:
                pose_metric[key] += pose_res[key]
            num_pcs += 1

    for key in pose_metric:
        pose_metric[key] /= max(num_pcs, 1)

    os.makedirs("./eval_result/ground_truth_pose_delft", exist_ok=True)
    os.makedirs("./eval_result/data_delft", exist_ok=True)

    for clip_info in test_loader.dataset.clips_info:
        gt_pose = [np.eye(4)]
        pred_pose = [np.eye(4)]
        gt_each = gt_trans_all[clip_info["index"][0]: clip_info["index"][1]].cpu().numpy()
        pred_each = pre_trans_all[clip_info["index"][0]: clip_info["index"][1]].cpu().numpy()

        for frame_id in range(gt_each.shape[0]):
            gt_pose.append(gt_pose[frame_id].dot(T_inv_function(gt_each[frame_id])))
            pred_pose.append(pred_pose[frame_id].dot(T_inv_function(pred_each[frame_id])))

        gt_pose = np.stack(gt_pose).reshape((len(gt_pose), -1))[:, :12]
        pred_pose = np.stack(pred_pose).reshape((len(pred_pose), -1))[:, :12]
        np.savetxt(f'./eval_result/ground_truth_pose_delft/{clip_info["clip_name"]}.txt', gt_pose, fmt="%f")
        np.savetxt(f'./eval_result/data_delft/{clip_info["clip_name"]}.txt', pred_pose, fmt="%f")

    infer_time = sum(infer_time_all) / max(len(infer_time_all), 1)
    textio.cprint("###The inference speed is %.4fms per frame###" % (infer_time * 1000))
    pose_metric_kitti = kittiOdomEval(split_path=args.split_path).eval(toCameraCoord=False)
    return pose_metric, pose_metric_kitti


def T_inv_function(matrix):
    rot = matrix[:3, :3]
    trans = matrix[:3, 3]
    inv = np.eye(4)
    inv[:3, :3] = np.linalg.inv(rot)
    inv[:3, 3] = -np.linalg.inv(rot).dot(trans)
    return inv
