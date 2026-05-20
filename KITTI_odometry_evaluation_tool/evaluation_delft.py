import glob
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml


def load_sequence_split(yaml_path: Path) -> Dict[str, List[str]]:
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Split file not found: {yaml_path}")
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return {k: [str(item) for item in v] for k, v in data.items()}


class kittiOdomEval:
    def __init__(self, split_path):
        self.split_path = Path(split_path)
        self.gt_dir = "./eval_result/ground_truth_pose_delft"
        self.result_dir = "./eval_result/data_delft"
        split_info = load_sequence_split(self.split_path)
        self.eval_seqs = split_info.get("test", split_info.get("val", []))
        if not self.eval_seqs:
            raise ValueError("No test/val sequences found in split file.")

        self.lengths = [20, 40, 60, 80, 100, 120, 140, 160]
        self.num_lengths = len(self.lengths)
        gt_files = glob.glob(self.gt_dir + "/*.txt")
        self.seqs_with_gt = [os.path.splitext(os.path.split(f)[1])[0] for f in gt_files]

    def loadPoses(self, file_name, toCameraCoord=False):
        poses = {}
        with open(file_name, "r") as f:
            for cnt, line in enumerate(f.readlines()):
                values = [float(i) for i in line.split()]
                has_idx = int(len(values) == 13)
                pose = np.eye(4)
                for row in range(3):
                    for col in range(4):
                        pose[row, col] = values[row * 4 + col + has_idx]
                frame_idx = values[0] if has_idx else cnt
                poses[frame_idx] = pose
        return poses

    def trajectoryDistances(self, poses):
        dist = [0.0]
        frame_ids = sorted(poses.keys())
        for i in range(len(frame_ids) - 1):
            p1 = poses[frame_ids[i]]
            p2 = poses[frame_ids[i + 1]]
            delta = p1[:3, 3] - p2[:3, 3]
            dist.append(dist[i] + np.linalg.norm(delta))
        self.distance = dist[-1]
        return dist

    def rotationError(self, pose_error):
        d = 0.5 * (pose_error[0, 0] + pose_error[1, 1] + pose_error[2, 2] - 1.0)
        return np.arccos(max(min(d, 1.0), -1.0))

    def translationError(self, pose_error):
        return np.linalg.norm(pose_error[:3, 3])

    def lastFrameFromSegmentLength(self, dist, first_frame, length):
        for i in range(first_frame, len(dist)):
            if dist[i] > dist[first_frame] + length:
                return i
        return -1

    def calcSequenceErrors(self, poses_gt, poses_result):
        errors = []
        dist = self.trajectoryDistances(poses_gt)
        step_size = 13
        for first_frame in range(0, len(poses_gt), step_size):
            for length in self.lengths:
                last_frame = self.lastFrameFromSegmentLength(dist, first_frame, length)
                if last_frame == -1:
                    continue
                if first_frame not in poses_result or last_frame not in poses_result:
                    continue
                pose_delta_gt = np.linalg.inv(poses_gt[first_frame]).dot(poses_gt[last_frame])
                pose_delta_pred = np.linalg.inv(poses_result[first_frame]).dot(poses_result[last_frame])
                pose_error = np.linalg.inv(pose_delta_pred).dot(pose_delta_gt)
                errors.append([
                    first_frame,
                    self.rotationError(pose_error) / length,
                    self.translationError(pose_error) / length,
                    length,
                ])
        return errors

    def computeOverallErr(self, seq_err):
        if len(seq_err) == 0:
            return 0.0, 0.0
        arr = np.asarray(seq_err)
        return float(np.mean(arr[:, 2])), float(np.mean(arr[:, 1]))

    def eval(self, toCameraCoord=False):
        ave_errs = {}
        os.makedirs(self.result_dir, exist_ok=True)
        for seq in self.eval_seqs:
            pred_file = os.path.join(self.result_dir, f"{seq}.txt")
            gt_file = os.path.join(self.gt_dir, f"{seq}.txt")
            if not os.path.exists(pred_file):
                raise FileNotFoundError(f"Prediction file not found: {pred_file}")
            if seq not in self.seqs_with_gt:
                continue
            poses_pred = self.loadPoses(pred_file, toCameraCoord=toCameraCoord)
            poses_gt = self.loadPoses(gt_file, toCameraCoord=False)
            seq_err = self.calcSequenceErrors(poses_gt, poses_pred)
            ave_t_err, ave_r_err = self.computeOverallErr(seq_err)
            print(f"\nSequence: {seq}")
            print("Distance (m): %d" % self.distance)
            print("Average sequence translational RMSE (%):   {0:.4f}".format(ave_t_err * 100))
            print("Average sequence rotational error (deg/m): {0:.4f}\n".format(ave_r_err / np.pi * 180))
            ave_errs[seq] = [ave_t_err, ave_r_err]

        if not ave_errs:
            return {"TRMSE": 0.0, "RRMSE": 0.0}

        trmse = np.mean([value[0] * 100 for value in ave_errs.values()])
        rrmse = np.mean([value[1] / np.pi * 180 for value in ave_errs.values()])
        return {"TRMSE": float(trmse), "RRMSE": float(rrmse)}
