import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


class RealtimeMetricsPlot:

    def __init__(self, save_dir="logs"):

        os.makedirs(save_dir, exist_ok=True)

        self.subplot_path = os.path.join(save_dir, "training_metrics_subplot.png")
        self.single_path = os.path.join(save_dir, "training_metrics_all.png")

        self.epochs = []

        self.train_loss = []
        self.val_loss = []
        self.dice = []
        self.iou = []
        self.precision = []
        self.recall = []

    def update(self, epoch, train_loss, val_loss, dice, iou, precision, recall):

        self.epochs.append(epoch)

        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)

        self.dice.append(dice)
        self.iou.append(iou)
        self.precision.append(precision)
        self.recall.append(recall)

        self.plot()

    def plot(self):

        # ========= 第一张图：6子图 =========
        plt.figure(figsize=(14,8))

        plt.subplot(2,3,1)
        plt.plot(self.epochs, self.train_loss)
        plt.title("Train Loss")

        plt.subplot(2,3,2)
        plt.plot(self.epochs, self.val_loss)
        plt.title("Val Loss")

        plt.subplot(2,3,3)
        plt.plot(self.epochs, self.dice)
        plt.title("Dice")

        plt.subplot(2,3,4)
        plt.plot(self.epochs, self.iou)
        plt.title("IoU")

        plt.subplot(2,3,5)
        plt.plot(self.epochs, self.precision)
        plt.title("Precision")

        plt.subplot(2,3,6)
        plt.plot(self.epochs, self.recall)
        plt.title("Recall")

        plt.tight_layout()
        plt.savefig(self.subplot_path)
        plt.close()


        # ========= 第二张图：所有曲线 =========
        plt.figure(figsize=(10,6))

        plt.plot(self.epochs, self.train_loss, label="Train Loss")
        plt.plot(self.epochs, self.val_loss, label="Val Loss")

        plt.plot(self.epochs, self.dice, label="Dice")
        plt.plot(self.epochs, self.iou, label="IoU")

        plt.plot(self.epochs, self.precision, label="Precision")
        plt.plot(self.epochs, self.recall, label="Recall")

        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Training Metrics")

        plt.legend()
        plt.grid(True)

        plt.savefig(self.single_path)
        plt.close()
        
        



class RealtimeMetricsPlot_edge:

    def __init__(self, save_dir="logs"):

        os.makedirs(save_dir, exist_ok=True)

        self.subplot_path = os.path.join(save_dir, "training_metrics_subplot.png")
        self.single_path = os.path.join(save_dir, "training_metrics_all.png")

        self.epochs = []

        self.train_loss = []
        self.val_loss = []
        self.val_seg_loss = []
        self.val_edge_loss = []
        self.dice = []
        self.iou = []
        self.precision = []
        self.recall = []

    def update(self, epoch, train_loss, val_loss, val_seg_loss, val_edge_loss, dice, iou, precision, recall):

        self.epochs.append(epoch)

        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)

        
        self.val_seg_loss.append(val_seg_loss)
        self.val_edge_loss.append(val_edge_loss)
        
        self.dice.append(dice)
        self.iou.append(iou)
        self.precision.append(precision)
        self.recall.append(recall)

        self.plot()


    def plot(self):

        plt.figure(figsize=(16,8))

        # ---------- Row 1 ----------
        plt.subplot(2,4,1)
        plt.plot(self.epochs, self.train_loss)
        plt.title("Train Loss")

        plt.subplot(2,4,2)
        plt.plot(self.epochs, self.val_loss)
        plt.title("Val Loss")

        plt.subplot(2,4,3)
        plt.plot(self.epochs, self.val_seg_loss)
        plt.title("Seg Loss")

        plt.subplot(2,4,4)
        plt.plot(self.epochs, self.val_edge_loss)
        plt.title("Edge Loss")

        # ---------- Row 2 ----------
        plt.subplot(2,4,5)
        plt.plot(self.epochs, self.dice)
        plt.title("Dice")

        plt.subplot(2,4,6)
        plt.plot(self.epochs, self.iou)
        plt.title("IoU")

        plt.subplot(2,4,7)
        plt.plot(self.epochs, self.precision)
        plt.title("Precision")

        plt.subplot(2,4,8)
        plt.plot(self.epochs, self.recall)
        plt.title("Recall")

        plt.tight_layout()
        plt.savefig(self.subplot_path)
        plt.close()
        
        
        # ========= 第二张图：所有曲线 =========
        plt.figure(figsize=(10,6))

        plt.plot(self.epochs, self.train_loss, label="Train Loss")
        plt.plot(self.epochs, self.val_loss, label="Val Loss")

        plt.plot(self.epochs, self.dice, label="Dice")
        plt.plot(self.epochs, self.iou, label="IoU")

        plt.plot(self.epochs, self.precision, label="Precision")
        plt.plot(self.epochs, self.recall, label="Recall")

        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Training Metrics")

        plt.legend()
        plt.grid(True)

        plt.savefig(self.single_path)
        plt.close()
        
        



class RealtimeMetricsPlot_edge_only:

    def __init__(self, save_dir="logs"):

        os.makedirs(save_dir, exist_ok=True)

        self.subplot_path = os.path.join(save_dir, "training_metrics_subplot.png")
        self.single_path = os.path.join(save_dir, "training_metrics_all.png")

        self.epochs = []

        self.train_loss = []
        self.val_loss = []
        self.edge_loss = []

    def update(self, epoch, train_loss, val_loss, edge_loss):

        self.epochs.append(epoch)

        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.edge_loss.append(edge_loss)

        self.plot()

    def plot(self):

        # ========= 子图 =========
        plt.figure(figsize=(12,4))

        plt.subplot(1,3,1)
        plt.plot(self.epochs, self.train_loss)
        plt.title("Train Loss")

        plt.subplot(1,3,2)
        plt.plot(self.epochs, self.val_loss)
        plt.title("Val Loss")

        plt.subplot(1,3,3)
        plt.plot(self.epochs, self.edge_loss)
        plt.title("Edge Loss")

        plt.tight_layout()
        plt.savefig(self.subplot_path)
        plt.close()

        # ========= 总图 =========
        plt.figure(figsize=(8,5))

        plt.plot(self.epochs, self.train_loss, label="Train Loss")
        plt.plot(self.epochs, self.val_loss, label="Val Loss")
        plt.plot(self.epochs, self.edge_loss, label="Edge Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Edge Detection Training")

        plt.legend()
        plt.grid(True)

        plt.savefig(self.single_path)
        plt.close()
        
        


class RealtimeMetricsPlot_edge2:

    def __init__(self, save_dir="logs"):

        os.makedirs(save_dir, exist_ok=True)

        self.subplot_path = os.path.join(save_dir, "training_metrics_subplot.png")
        self.single_path = os.path.join(save_dir, "training_metrics_all.png")

        self.epochs = []

        # ===== train =====
        self.train_loss = []
        self.train_seg_loss = []
        self.train_edge_loss = []

        # ===== val =====
        self.val_loss = []
        self.val_seg_loss = []
        self.val_edge_loss = []

        # ===== metrics =====
        self.dice = []
        self.iou = []
        self.precision = []
        self.recall = []

    def update(
        self,
        epoch,

        # train
        train_loss,
        val_loss,

        train_seg_loss,
        train_edge_loss,

        # val
        val_seg_loss,
        val_edge_loss,

        # metrics
        dice,
        iou,
        precision,
        recall
    ):

        self.epochs.append(epoch)

        # ===== train =====
        self.train_loss.append(train_loss)
        self.train_seg_loss.append(train_seg_loss)
        self.train_edge_loss.append(train_edge_loss)

        # ===== val =====
        self.val_loss.append(val_loss)
        self.val_seg_loss.append(val_seg_loss)
        self.val_edge_loss.append(val_edge_loss)

        # ===== metrics =====
        self.dice.append(dice)
        self.iou.append(iou)
        self.precision.append(precision)
        self.recall.append(recall)

        self.plot()

    def plot(self):

        # ================= 子图 =================
        plt.figure(figsize=(16,8))

        # ---------- Row 1 ----------
        plt.subplot(2,4,1)
        plt.plot(self.epochs, self.train_loss)
        plt.title("Train Total Loss")

        plt.subplot(2,4,2)
        plt.plot(self.epochs, self.val_loss)
        plt.title("Val Total Loss")

        plt.subplot(2,4,3)
        plt.plot(self.epochs, self.train_seg_loss)
        plt.title("Train Seg Loss")

        plt.subplot(2,4,4)
        plt.plot(self.epochs, self.val_seg_loss)
        plt.title("Val Seg Loss")

        # ---------- Row 2 ----------
        plt.subplot(2,4,5)
        plt.plot(self.epochs, self.train_edge_loss)
        plt.title("Train Edge Loss")

        plt.subplot(2,4,6)
        plt.plot(self.epochs, self.val_edge_loss)
        plt.title("Val Edge Loss")

        plt.subplot(2,4,7)
        plt.plot(self.epochs, self.dice)
        plt.title("Dice")

        plt.subplot(2,4,8)
        plt.plot(self.epochs, self.iou)
        plt.title("IoU")

        plt.tight_layout()
        plt.savefig(self.subplot_path)
        plt.close()

        # ================= 总图 =================
        plt.figure(figsize=(10,6))

        plt.plot(self.epochs, self.train_loss, label="Train Total")
        plt.plot(self.epochs, self.val_loss, label="Val Total")

        plt.plot(self.epochs, self.train_seg_loss, label="Train Seg")
        plt.plot(self.epochs, self.val_seg_loss, label="Val Seg")

        plt.plot(self.epochs, self.train_edge_loss, label="Train Edge")
        plt.plot(self.epochs, self.val_edge_loss, label="Val Edge")

        plt.plot(self.epochs, self.dice, label="Dice")
        plt.plot(self.epochs, self.iou, label="IoU")

        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Training Metrics")

        plt.legend()
        plt.grid(True)

        plt.savefig(self.single_path)
        plt.close()
        
        
        
class RealtimeMetricsPlot_MTL:

    def __init__(self, save_dir="logs"):

        os.makedirs(save_dir, exist_ok=True)

        self.save_path = os.path.join(save_dir, "training_curve.png")

        self.epochs = []

        self.train_loss = []
        self.val_loss = []

        self.seg_loss = []
        self.edge_loss = []

        self.dice = []
        self.iou = []
        self.precision = []
        self.recall = []

    def update(
        self,
        epoch,
        train_loss,
        val_loss,
        seg_loss,
        edge_loss,
        dice,
        iou,
        precision,
        recall
    ):

        self.epochs.append(epoch)

        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)

        self.seg_loss.append(seg_loss)
        self.edge_loss.append(edge_loss)

        self.dice.append(dice)
        self.iou.append(iou)
        self.precision.append(precision)
        self.recall.append(recall)

        self.plot()

    def plot(self):

        import matplotlib.pyplot as plt

        plt.figure(figsize=(14,6))

        # ===== loss =====
        plt.subplot(1,2,1)
        plt.plot(self.epochs, self.train_loss, label="Train")
        plt.plot(self.epochs, self.val_loss, label="Val")
        plt.plot(self.epochs, self.seg_loss, label="Seg")
        plt.plot(self.epochs, self.edge_loss, label="Edge")
        plt.title("Loss")
        plt.legend()

        # ===== metrics =====
        plt.subplot(1,2,2)
        plt.plot(self.epochs, self.dice, label="Dice")
        plt.plot(self.epochs, self.iou, label="IoU")
        plt.plot(self.epochs, self.precision, label="Precision")
        plt.plot(self.epochs, self.recall, label="Recall")
        plt.title("Metrics")
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.save_path)
        plt.close()
        
        


class RealtimeMetricsPlot_MTL_3:

    def __init__(self, save_dir="logs"):

        os.makedirs(save_dir, exist_ok=True)
        self.save_path = os.path.join(save_dir, "training_curve.png")

        self.epochs = []

        # ===== total =====
        self.train_loss = []
        self.val_loss = []

        # ===== task loss =====
        self.val_seg = []
        self.val_edge = []
        self.val_cls = []   # ⭐ 新增

        # ===== metrics =====
        self.dice = []
        self.iou = []
        self.precision = []
        self.recall = []

    def update(
        self,
        epoch,
        train_loss,
        val_loss,
        val_seg,
        val_edge,
        val_cls,   # ⭐ 新增
        dice,
        iou,
        precision,
        recall
    ):

        self.epochs.append(epoch)

        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)

        self.val_seg.append(val_seg)
        self.val_edge.append(val_edge)
        self.val_cls.append(val_cls)   # ⭐ 新增

        self.dice.append(dice)
        self.iou.append(iou)
        self.precision.append(precision)
        self.recall.append(recall)

        self.plot()

    def plot(self):

        plt.figure(figsize=(14,8))

        # ===== Total Loss =====
        plt.subplot(2,2,1)
        plt.plot(self.epochs, self.train_loss, label="Train Total")
        plt.plot(self.epochs, self.val_loss, label="Val Total")
        plt.title("Total Loss")
        plt.legend()

        # ===== Task Loss =====
        plt.subplot(2,2,2)
        plt.plot(self.epochs, self.val_seg, label="Seg")
        plt.plot(self.epochs, self.val_edge, label="Edge")
        plt.plot(self.epochs, self.val_cls, label="CLS", linestyle='--')  # ⭐ 新增
        plt.title("Task Loss (Val)")
        plt.legend()

        # ===== Metrics =====
        plt.subplot(2,2,3)
        plt.plot(self.epochs, self.dice, label="Dice")
        plt.plot(self.epochs, self.iou, label="IoU")
        plt.title("Overlap Metrics")
        plt.legend()

        plt.subplot(2,2,4)
        plt.plot(self.epochs, self.precision, label="Precision")
        plt.plot(self.epochs, self.recall, label="Recall")
        plt.title("Precision / Recall")
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.save_path)
        plt.close()