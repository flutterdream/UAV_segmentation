import numpy as np
import torch

from .labels import CLASS_NAMES, IGNORE_TRAIN_ID, NUM_CLASSES


class SegmentationMetrics:
    def __init__(self, num_classes=NUM_CLASSES):
        self.num_classes = num_classes
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    @torch.no_grad()
    def update(self, logits, target):
        prediction = logits.argmax(dim=1)
        valid = target != IGNORE_TRAIN_ID
        target = target[valid].to(torch.int64)
        prediction = prediction[valid].to(torch.int64)
        encoded = target * self.num_classes + prediction
        counts = torch.bincount(
            encoded, minlength=self.num_classes * self.num_classes
        )
        self.confusion += counts.reshape(self.num_classes, self.num_classes).cpu().numpy()

    def compute(self):
        true_positive = np.diag(self.confusion).astype(np.float64)
        false_positive = self.confusion.sum(axis=0) - true_positive
        false_negative = self.confusion.sum(axis=1) - true_positive
        denominator = true_positive + false_positive + false_negative
        iou = np.divide(
            true_positive,
            denominator,
            out=np.full_like(true_positive, np.nan),
            where=denominator > 0,
        )
        return {
            "miou": float(np.nanmean(iou)),
            "per_class_iou": {
                name: None if np.isnan(value) else float(value)
                for name, value in zip(CLASS_NAMES, iou)
            }
        }

