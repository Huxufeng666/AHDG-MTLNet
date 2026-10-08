import os
import csv


class TrainLogger_Text_MTL:
    def __init__(self, log_dir):
        self.log_path = os.path.join(log_dir, "train_log.csv")

        # 表头（论文级）
        self.header = [
            "epoch",
            "train_total_loss",
            "val_total_loss",

            "train_seg_loss",
            "train_edge_loss",
            "train_cls_loss",

            "val_seg_loss",
            "val_edge_loss",
            "val_cls_loss",

            "val_dice",
            "val_iou",
            "val_precision",
            "val_recall",

            "lr"
        ]

        # 写表头
        with open(self.log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.header)

    def log(
        self,
        epoch,
        train_total,
        val_total,

        train_seg,
        train_edge,
        train_cls,

        val_seg,
        val_edge,
        val_cls,

        val_dice,
        val_iou,
        val_precision,
        val_recall,

        lr
    ):
        row = [
            epoch,
            train_total,
            val_total,

            train_seg,
            train_edge,
            train_cls,

            val_seg,
            val_edge,
            val_cls,

            val_dice,
            val_iou,
            val_precision,
            val_recall,

            lr
        ]

        with open(self.log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)