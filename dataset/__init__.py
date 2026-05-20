from .vod_clip_odm import vodClipDatasetOdm

dataset_dict = {
    "vodClipDatasetOdm": vodClipDatasetOdm,
    "scClipDatasetOdm": vodClipDatasetOdm,
}

__all__ = ["vodClipDatasetOdm", "dataset_dict"]
