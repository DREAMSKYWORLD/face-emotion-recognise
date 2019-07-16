from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np

class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, startAt=0):
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.startAt = startAt

    def on_train_begin(self, logs={}):
        # 初始化历史
        self.H = {}

    def on_epoch_end(self, epoch, logs={}):
        # 循环记录训练损失、准确性
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l

        # 两个epoch后开始绘制图形
        if len(self.H["loss"]) > 1:
            # 绘制训练历史
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"], label="train_loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.plot(N, self.H["acc"], label="train_acc")
            plt.plot(N, self.H["val_acc"], label="val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(
                len(self.H["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()

            # 保存图片
            plt.savefig(self.figPath)
            plt.close()
