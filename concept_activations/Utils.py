import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, f1_score, recall_score


class Utils:

    def imshow(img):
        # img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    @staticmethod
    def is_cuda_available():
        return torch.cuda.is_available()

    @staticmethod
    def get_device():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def crop_center(img, cropx, cropy):

        if len(img.shape) == 3:
            img = img[:, :, 0]
        try:
            y, x = img.shape
        except:
            pdb.set_trace()
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty:starty + cropy, startx:startx + cropx]

    @staticmethod
    def save_model(epoch_id,
                   loss_train,
                   loss_val,
                   loss_min,
                   model_state_dict,
                   optimizer_state_dict,
                   chk_pt_path):
        torch.save({
            "epoch": epoch_id,
            "train_loss": loss_train,
            "val_loss": loss_val,
            "loss_min": loss_min,
            "state_dict": model_state_dict,
            "optimizer": optimizer_state_dict
        }, chk_pt_path)

    @staticmethod
    def compute_AUROC_recall(output_GT, output_pred, output_size):
        out_AUROC = []
        out_recall = []
        data_GT = output_GT.cpu().numpy()
        data_PRED = output_pred.cpu().numpy()

        for i in range(output_size):
            try:
                out_AUROC.append(
                    roc_auc_score(data_GT[:, i],
                                  data_PRED[:, i]))

                data_PRED[:, i] = 1/(1 + np.exp(-data_PRED[:, i]))
                pred_y = (data_PRED[:, i] > 0.53).astype(int)
                out_recall.append(
                    recall_score(data_GT[:, i],
                                 pred_y))
            except ValueError:
                pass
        return out_AUROC, out_recall

    @staticmethod
    def compute_F1(output_GT, output_pred, output_size):
        F1 = []
        data_GT = output_GT.cpu().numpy()
        data_PRED = output_pred.cpu().numpy()

        for i in range(output_size):
            try:
                F1.append(f1_score(data_GT[:, i].tolist(),
                                   data_PRED[:, i].tolist(),
                                   average=None,
                                   zero_division=1))
            except ValueError:
                pass
        return F1

    @staticmethod
    def compute_AUROC_numpy(output_GT, output_pred, output_size):
        out_AUROC = []

        for i in range(output_size):
            try:
                out_AUROC.append(
                    roc_auc_score(output_GT[:, i],
                                  output_pred[:, i]))
            except ValueError:
                pass
        return out_AUROC

    @staticmethod
    def compute_F1_numpy(output_GT, output_pred, output_size):
        F1 = []

        for i in range(output_size):
            try:
                F1.append(f1_score(output_GT[:, i].tolist(),
                                   output_pred[:, i].tolist(),
                                   average=None))
            except ValueError:
                pass
        return F1

    @staticmethod
    def read_csv(path):
        return pd.read_csv(path)
