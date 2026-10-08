import os
import csv
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter


class TrainLogger:

    def __init__(self, save_dir="logs"):

        os.makedirs(save_dir, exist_ok=True)

        self.save_dir = save_dir
        self.csv_file = os.path.join(save_dir, "train_log.csv")


        # CSV header
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "avg_seg_loss",
                "avg_edge_loss",
                "avg_val_dice",
                "iou",
                "precision",
                "recall",
                "lr"
            ])

        # 保存数据用于画图
        self.epochs = []
        self.train_losses = []
        self.val_losses = []

    def log(self, epoch, train_loss, val_loss, dice, iou, precision, recall, lr):

        # CSV记录
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                round(train_loss, 6),
                round(val_loss, 6),
                round(dice, 6),
                round(iou, 6),
                round(precision, 6),
                round(recall, 6),
                lr
            ])

    
        # 保存画图数据
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

    def plot_loss(self):

        plt.figure()

        plt.plot(self.epochs, self.train_losses, label="Train Loss")
        plt.plot(self.epochs, self.val_losses, label="Val Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.savefig(os.path.join(self.save_dir, "loss_curve.png"))
        plt.close()

    def close(self):
        self.tb_writer.close()
        
        


class TrainLogger_edge:

    def __init__(self, save_dir="logs"):

        os.makedirs(save_dir, exist_ok=True)

        self.save_dir = save_dir
        self.csv_file = os.path.join(save_dir, "train_log.csv")

        # CSV header
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "avg_seg_loss",
                "avg_edge_loss",
                "avg_val_dice",
                "iou",
                "precision",
                "recall",
                "lr"
            ])

        # 保存数据用于画图
        self.epochs = []
        self.train_losses = []
        self.val_losses = []

    def log(
        self,
        epoch,
        train_loss,
        val_loss,
        avg_seg_loss,
        avg_edge_loss,
        avg_val_dice,
        iou,
        precision,
        recall,
        lr
    ):

        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                epoch,
                round(train_loss, 6),
                round(val_loss, 6),
                round(avg_seg_loss, 6),
                round(avg_edge_loss, 6),
                round(avg_val_dice, 6),
                round(iou, 6),
                round(precision, 6),
                round(recall, 6),
                lr
            ])
            


class TrainLogger_edge_only:

    def __init__(self, save_dir="logs"):

        os.makedirs(save_dir, exist_ok=True)

        self.save_dir = save_dir
        self.csv_file = os.path.join(save_dir, "train_log.csv")

        # CSV header（简化版）
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "edge_loss",
                "lr"
            ])

        # 用于画图
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.edge_losses = []

    def log(
        self,
        epoch,
        train_loss,
        val_loss,
        edge_loss,
        lr
    ):

        # 保存到内存（用于画图）
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.edge_losses.append(edge_loss)

        # 写CSV
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                epoch,
                round(train_loss, 6),
                round(val_loss, 6),
                round(edge_loss, 6),
                lr
            ])
            
            


   
   
        
class TrainLogger_edge2:

    def __init__(self, save_dir="logs"):
        os.makedirs(save_dir, exist_ok=True)

        self.csv_file = os.path.join(save_dir, "train_log.csv")

        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "seg_loss",
                "edge_loss",
                "dice",
                "iou",
                "precision",
                "recall",
                "lr"
            ])

    def log(
        self,
        epoch,
        train_loss,
        val_loss,
        seg_loss,
        edge_loss,
        dice,
        iou,
        precision,
        recall,
        lr
    ):
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                round(train_loss, 6),
                round(val_loss, 6),
                round(seg_loss, 6),
                round(edge_loss, 6),
                round(dice, 6),
                round(iou, 6),
                round(precision, 6),
                round(recall, 6),
                lr
            ])
            



class TrainLogger_MTL():

    def __init__(self, save_dir="log"):

        os.makedirs(save_dir, exist_ok=True)

        self.save_dir = save_dir
        self.csv_file = os.path.join(save_dir, "train_log.csv")
        self.plot_path = os.path.join(save_dir, "training_curve.png")

        # ===== CSV header =====
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",

                "train_total",
                "train_seg",
                "train_edge",

                "val_total",
                "val_seg",
                "val_edge",

                "dice",
                "iou",
                "precision",
                "recall",
                "lr"
            ])

        # ===== 存数据用于画图 =====
        self.epochs = []

        self.train_total = []
        self.train_seg = []
        self.train_edge = []

        self.val_total = []
        self.val_seg = []
        self.val_edge = []

        self.dice = []
        self.iou = []
        self.precision = []
        self.recall = []

    def log(
        self,
        epoch,

        train_total,
        val_total,

        train_seg,
        train_edge,

        val_seg,
        val_edge,

        dice,
        iou,
        precision,
        recall,
        lr
    ):

        # ===== 写 CSV =====
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                epoch,

                round(train_total, 6),
                round(train_seg, 6),
                round(train_edge, 6),

                round(val_total, 6),
                round(val_seg, 6),
                round(val_edge, 6),

                round(dice, 6),
                round(iou, 6),
                round(precision, 6),
                round(recall, 6),

                lr
            ])

        # ===== 存数据 =====
        self.epochs.append(epoch)

        self.train_total.append(train_total)
        self.train_seg.append(train_seg)
        self.train_edge.append(train_edge)

        self.val_total.append(val_total)
        self.val_seg.append(val_seg)
        self.val_edge.append(val_edge)

        self.dice.append(dice)
        self.iou.append(iou)
        self.precision.append(precision)
        self.recall.append(recall)

        # ===== 自动画图 =====
        self.plot()

    def plot(self):

        plt.figure(figsize=(14,8))

        # ===== Loss =====
        plt.subplot(2,2,1)
        plt.plot(self.epochs, self.train_total, label="Train Total")
        plt.plot(self.epochs, self.val_total, label="Val Total")
        plt.title("Total Loss")
        plt.legend()

        plt.subplot(2,2,2)
        plt.plot(self.epochs, self.train_seg, label="Train Seg")
        plt.plot(self.epochs, self.val_seg, label="Val Seg")
        plt.title("Seg Loss")
        plt.legend()

        plt.subplot(2,2,3)
        plt.plot(self.epochs, self.train_edge, label="Train Edge")
        plt.plot(self.epochs, self.val_edge, label="Val Edge")
        plt.title("Edge Loss")
        plt.legend()

        # ===== Metrics =====
        plt.subplot(2,2,4)
        plt.plot(self.epochs, self.dice, label="Dice")
        plt.plot(self.epochs, self.iou, label="IoU")
        plt.plot(self.epochs, self.precision, label="Precision")
        plt.plot(self.epochs, self.recall, label="Recall")
        plt.title("Metrics")
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.plot_path)
        plt.close()
        
        




class TrainLogger_MTL_3():

    def __init__(self, save_dir="log"):

        os.makedirs(save_dir, exist_ok=True)

        self.save_dir = save_dir
        self.csv_file = os.path.join(save_dir, "train_log.csv")
        self.plot_path = os.path.join(save_dir, "training_curve.png")

        # ===== CSV header =====
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",

                "train_total",
                "train_seg",
                "train_edge",
                "train_cls",   # ⭐ 新增

                "val_total",
                "val_seg",
                "val_edge",
                "val_cls",     # ⭐ 新增

                "dice",
                "iou",
                "precision",
                "recall",
                "lr"
            ])

        # ===== 存数据 =====
        self.epochs = []

        self.train_total = []
        self.train_seg = []
        self.train_edge = []
        self.train_cls = []   # ⭐ 新增

        self.val_total = []
        self.val_seg = []
        self.val_edge = []
        self.val_cls = []     # ⭐ 新增

        self.dice = []
        self.iou = []
        self.precision = []
        self.recall = []

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

        dice,
        iou,
        precision,
        recall,
        lr
    ):

        # ===== 写 CSV =====
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                epoch,

                round(train_total, 6),
                round(train_seg, 6),
                round(train_edge, 6),
                round(train_cls, 6),   # ⭐ 新增

                round(val_total, 6),
                round(val_seg, 6),
                round(val_edge, 6),
                round(val_cls, 6),     # ⭐ 新增

                round(dice, 6),
                round(iou, 6),
                round(precision, 6),
                round(recall, 6),

                lr
            ])

        # ===== 存数据 =====
        self.epochs.append(epoch)

        self.train_total.append(train_total)
        self.train_seg.append(train_seg)
        self.train_edge.append(train_edge)
        self.train_cls.append(train_cls)   # ⭐ 新增

        self.val_total.append(val_total)
        self.val_seg.append(val_seg)
        self.val_edge.append(val_edge)
        self.val_cls.append(val_cls)       # ⭐ 新增

        self.dice.append(dice)
        self.iou.append(iou)
        self.precision.append(precision)
        self.recall.append(recall)

        self.plot()

    # def plot(self):

    #     plt.figure(figsize=(14,10))

    #     # ===== Total Loss =====
    #     plt.subplot(2,2,1)
    #     plt.plot(self.epochs, self.train_total, label="Train Total")
    #     plt.plot(self.epochs, self.val_total, label="Val Total")
    #     plt.title("Total Loss")
    #     plt.legend()

    #     # ===== Seg Loss =====
    #     plt.subplot(2,2,2)
    #     plt.plot(self.epochs, self.train_seg, label="Train Seg")
    #     plt.plot(self.epochs, self.val_seg, label="Val Seg")
    #     plt.title("Seg Loss")
    #     plt.legend()

    #     # ===== Edge + CLS ===== ⭐ 改这里
    #     plt.subplot(2,2,3)
    #     plt.plot(self.epochs, self.train_edge, label="Train Edge")
    #     plt.plot(self.epochs, self.val_edge, label="Val Edge")

    #     plt.plot(self.epochs, self.train_cls, '--', label="Train CLS")
    #     plt.plot(self.epochs, self.val_cls, '--', label="Val CLS")

    #     plt.title("Edge & CLS Loss")
    #     plt.legend()

    #     # ===== Metrics =====
    #     plt.subplot(2,2,4)
    #     plt.plot(self.epochs, self.dice, label="Dice")
    #     plt.plot(self.epochs, self.iou, label="IoU")
    #     plt.plot(self.epochs, self.precision, label="Precision")
    #     plt.plot(self.epochs, self.recall, label="Recall")
    #     plt.title("Metrics")
    #     plt.legend()

    #     plt.tight_layout()
    #     plt.savefig(self.plot_path)
    #     plt.close()
    def plot(self):

        plt.figure(figsize=(14,12))

        # ===== 1. Total Loss =====
        plt.subplot(3,2,1)
        plt.plot(self.epochs, self.train_total, label="Train Total")
        plt.plot(self.epochs, self.val_total, label="Val Total")
        plt.title("Total Loss")
        plt.legend()

        # ===== 2. Seg Loss =====
        plt.subplot(3,2,2)
        plt.plot(self.epochs, self.train_seg, label="Train Seg")
        plt.plot(self.epochs, self.val_seg, label="Val Seg")
        plt.title("Seg Loss")
        plt.legend()

        # ===== 3. Edge Loss =====
        plt.subplot(3,2,3)
        plt.plot(self.epochs, self.train_edge, label="Train Edge")
        plt.plot(self.epochs, self.val_edge, label="Val Edge")
        plt.title("Edge Loss")
        plt.legend()

        # ===== 4. CLS Loss（🔥 单独出来）=====
        plt.subplot(3,2,4)
        plt.plot(self.epochs, self.train_cls,  label="Train CLS")
        plt.plot(self.epochs, self.val_cls,  label="Val CLS")
        plt.title("CLS Loss")
        plt.legend()

        # ===== 5. Metrics（占一整行）=====
        plt.subplot(3,1,3)
        plt.plot(self.epochs, self.dice, label="Dice")
        plt.plot(self.epochs, self.iou, label="IoU")
        plt.plot(self.epochs, self.precision, label="Precision")
        plt.plot(self.epochs, self.recall, label="Recall")
        plt.title("Metrics")
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.plot_path)
        plt.close()