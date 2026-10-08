import os
import matplotlib.pyplot as plt


class RealtimeMetricsPlot_Text_MTL:
    def __init__(self, log_dir):
        self.log_dir = log_dir

        self.epochs = []

        self.train_loss = []
        self.val_loss = []

        self.seg_loss = []
        self.edge_loss = []
        self.cls_loss = []

        self.dice = []
        self.iou = []
        self.precision = []
        self.recall = []

        os.makedirs(os.path.join(log_dir, "plots"), exist_ok=True)

    def update(
        self,
        epoch,
        train_total,
        val_total,

        val_seg,
        val_edge,
        val_cls,

        val_dice,
        val_iou,
        val_precision,
        val_recall
    ):
        self.epochs.append(epoch)

        self.train_loss.append(train_total)
        self.val_loss.append(val_total)

        self.seg_loss.append(val_seg)
        self.edge_loss.append(val_edge)
        self.cls_loss.append(val_cls)

        self.dice.append(val_dice)
        self.iou.append(val_iou)
        self.precision.append(val_precision)
        self.recall.append(val_recall)

        self.plot_all()

    def plot_all(self):
        self.plot_loss()
        self.plot_tasks()
        self.plot_metrics()

    def plot_loss(self):
        plt.figure()
        plt.plot(self.epochs, self.train_loss, label="Train Loss")
        plt.plot(self.epochs, self.val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()

        save_path = os.path.join(self.log_dir, "plots", "loss_curve.png")
        plt.savefig(save_path)
        plt.close()

    def plot_tasks(self):
        plt.figure()
        plt.plot(self.epochs, self.seg_loss, label="Seg Loss")
        plt.plot(self.epochs, self.edge_loss, label="Edge Loss")
        plt.plot(self.epochs, self.cls_loss, label="Cls Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Task Loss")
        plt.legend()
        plt.grid()

        save_path = os.path.join(self.log_dir, "plots", "task_loss_curve.png")
        plt.savefig(save_path)
        plt.close()

    def plot_metrics(self):
        plt.figure()
        plt.plot(self.epochs, self.dice, label="Dice")
        plt.plot(self.epochs, self.iou, label="IoU")
        plt.plot(self.epochs, self.precision, label="Precision")
        plt.plot(self.epochs, self.recall, label="Recall")
        plt.xlabel("Epoch")
        plt.ylabel("Metrics")
        plt.legend()
        plt.grid()

        save_path = os.path.join(self.log_dir, "plots", "metrics_curve.png")
        plt.savefig(save_path)
        plt.close()