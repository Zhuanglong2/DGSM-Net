from .radar_loss import RadarOdmLoss

loss_dict = {
    "pgsm": {
        "odmLoss": [],
        "tLoss": [],
        "qLoss": [],
    }
}

__all__ = ["RadarOdmLoss", "loss_dict"]
