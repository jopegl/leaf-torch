import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)
        self.area_abs_error_sum = 0.
        self.area_pixel_count = 0.
        self.rer_leaf = []
        self.rer_marker = []

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.area_abs_error_sum = 0.
        self.area_pixel_count = 0.
        self.rer_leaf = []
        self.rer_marker = []


    def add_area_batch(self, pred_area, target_area):
        # Normaliza dimensões
        if pred_area.dim() == 3:
            pred_area = pred_area.unsqueeze(1)
        if target_area.dim() == 3:
            target_area = target_area.unsqueeze(1)

        # Match spatial size
        pred_area = F.interpolate(pred_area, size=target_area.shape[2:], mode='bilinear', align_corners=False)

        # Máscara: conta só regiões de folha/quadrado
        mask = target_area > 0

        # Soma erro absoluto só onde há folha
        abs_error = torch.abs(pred_area - target_area) * mask

        self.area_abs_error_sum += abs_error.sum().item()
        self.area_pixel_count += mask.sum().item()

    def area_rer_stats(self):
        return {
            "leaf_mean": np.mean(self.rer_leaf) if len(self.rer_leaf) else 0.0,
            "leaf_std":  np.std(self.rer_leaf)  if len(self.rer_leaf) else 0.0,
            "marker_mean": np.mean(self.rer_marker) if len(self.rer_marker) else 0.0,
            "marker_std":  np.std(self.rer_marker)  if len(self.rer_marker) else 0.0,
        }

    
    def area_rer_batch(self, final_area_pred, area_target, target_mask):
        B = final_area_pred.shape[0]

        for i in range(B):

            # -------- FOLHA --------
            leaf_mask = (target_mask[i] == 1)
            A_est_leaf = final_area_pred[i,0][leaf_mask].sum().item()
            A_gt_leaf  = area_target[i,0][leaf_mask].sum().item()

            if A_gt_leaf > 0:
                rer_leaf = abs(A_est_leaf - A_gt_leaf) / A_gt_leaf
                self.rer_leaf.append(rer_leaf)

            # -------- MARCADOR --------
            marker_mask = (target_mask[i] == 2)
            A_est_marker = final_area_pred[i,0][marker_mask].sum().item()
            A_gt_marker  = area_target[i,0][marker_mask].sum().item()

            if A_gt_marker > 0:
                rer_marker = abs(A_est_marker - A_gt_marker) / A_gt_marker
                self.rer_marker.append(rer_marker)

    def area_accuracy(self):
        if self.area_pixel_count == 0:
            return 0.0
        area_mae = self.area_abs_error_sum / self.area_pixel_count
        area_accuracy = 1.0 - area_mae
        return area_accuracy