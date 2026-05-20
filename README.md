# PGSM-Net

This repository contains a clean release version of PGSM-Net for 4D radar odometry. It keeps only the proposed model and the minimal training/evaluation pipeline used for the paper experiments.

## What Is Included

- `models/pgsm_net.py`: PGSM-Net model.
- `dataset/vod_clip_odm.py`: dataset loader for preprocessed odometry samples.
- `losses/radar_loss.py`: odometry loss.
- `engine.py` and `main.py`: training and evaluation loops.
- `KITTI_odometry_evaluation_tool/`: lightweight odometry evaluation for `t_rel` and `r_rel`.
- `splits/`: example split files for odometry experiments.

Visualization scripts, raw-data preprocessing scripts, and baseline model implementations are intentionally removed to keep the release focused and easy to reproduce.

## Environment

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The code also requires PyTorch, CUDA, and `pointnet2_ops`. Please install the versions compatible with your CUDA environment.

## Checkpoints and Datasets

For the anonymous review stage, we provide the pretrained checkpoints and preprocessed datasets for the VoD and SC experiments:

- [Checkpoints](https://pan.baidu.com/s/1vsFVfBvVqKTAN-RMqD41HQ?pwd=1234), extraction code: `1234`
- [Preprocessed datasets](https://pan.baidu.com/s/1pM-dVXleM2u2t01UXjKz8Q?pwd=1234), extraction code: `1234`

After downloading, update `dataset_path` and `model_path` in `configs.yaml` according to your local paths.

## Data Format

The release code expects preprocessed odometry samples, not raw radar data. Set `dataset_path` in `configs.yaml` to a directory with the following structure:

```text
/path/to/preprocessed/odm_smp/
  train/
    clip_0/
      000000.json
      000001.json
      ...
  test/
    clip_1/
      000000.json
      000001.json
      ...
```

Each JSON sample should contain:

- `pc1`: source radar points, with columns `[x, y, z, rcs, doppler, ...]`.
- `pc2`: target radar points, with columns `[x, y, z, rcs, doppler, ...]`.
- `trans`: the relative transformation matrix between the two frames.

## Training

Edit `configs.yaml`, especially `dataset_path`, `split_path`, and `cuda_device`, then run:

```bash
python main.py --config configs.yaml
```

The best checkpoint is saved to:

```text
checkpoints/pgsm_net/models/model.best.t7
```

For VoD experiments, use:

```yaml
dataset: vodClipDatasetOdm
split_path: splits/vod_odometry_split_info.yaml
```

For SC experiments, use:

```yaml
dataset: scClipDatasetOdm
split_path: splits/sc_odometry_split_info.yaml
```

## Evaluation

Set `model_path` in `configs.yaml`, or place the checkpoint at the default path above. Then run:

```bash
python main.py --config configs.yaml --eval
```

The script reports per-sequence and average odometry metrics, including translational RMSE and rotational RMSE.
