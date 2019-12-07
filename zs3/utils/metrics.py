import numpy as np


class Evaluator:
    def __init__(self, num_class, seen_classes_idx=None, unseen_classes_idx=None):
        self.num_class = num_class
        self.seen_classes_idx = seen_classes_idx
        self.unseen_classes_idx = unseen_classes_idx
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        if self.seen_classes_idx and self.unseen_classes_idx:
            Acc_seen = (
                np.diag(self.confusion_matrix)[self.seen_classes_idx].sum()
                / self.confusion_matrix[self.seen_classes_idx, :].sum()
            )
            Acc_unseen = (
                np.diag(self.confusion_matrix)[self.unseen_classes_idx].sum()
                / self.confusion_matrix[self.unseen_classes_idx, :].sum()
            )
            return Acc, Acc_seen, Acc_unseen
        else:
            return Acc

    def Pixel_Accuracy_Class(self):
        Acc_by_class = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(np.nan_to_num(Acc_by_class))
        if self.seen_classes_idx and self.unseen_classes_idx:
            Acc_seen = np.nanmean(np.nan_to_num(Acc_by_class[self.seen_classes_idx]))
            Acc_unseen = np.nanmean(np.nan_to_num(Acc_by_class[self.unseen_classes_idx]))
            return Acc, Acc_by_class, Acc_seen, Acc_unseen
        else:
            return Acc, Acc_by_class

    def Mean_Intersection_over_Union(self):
        MIoU_by_class = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1)
            + np.sum(self.confusion_matrix, axis=0)
            - np.diag(self.confusion_matrix)
        )
        MIoU = np.nanmean(np.nan_to_num(MIoU_by_class))
        if self.seen_classes_idx and self.unseen_classes_idx:
            MIoU_seen = np.nanmean(np.nan_to_num(MIoU_by_class[self.seen_classes_idx]))
            MIoU_unseen = np.nanmean(
                np.nan_to_num(MIoU_by_class[self.unseen_classes_idx])
            )
            return MIoU, MIoU_by_class, MIoU_seen, MIoU_unseen
        else:
            return MIoU, MIoU_by_class

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1)
            + np.sum(self.confusion_matrix, axis=0)
            - np.diag(self.confusion_matrix)
        )
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        if self.seen_classes_idx and self.unseen_classes_idx:
            FWIoU_seen = (
                freq[self.seen_classes_idx][freq[self.seen_classes_idx] > 0]
                * iu[self.seen_classes_idx][freq[self.seen_classes_idx] > 0]
            ).sum()
            FWIoU_unseen = (
                freq[self.unseen_classes_idx][freq[self.unseen_classes_idx] > 0]
                * iu[self.unseen_classes_idx][freq[self.unseen_classes_idx] > 0]
            ).sum()
            return FWIoU, FWIoU_seen, FWIoU_unseen
        else:
            return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype("int") + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


class Evaluator_seen_unseen:
    def __init__(self, num_class, unseen_classes_idx):
        self.num_class = num_class
        self.unseen_classes_idx = unseen_classes_idx

    def _fast_hist(self, label_true, label_pred, n_class, target="all", unseen=None):
        mask = (label_true >= 0) & (label_true < n_class)

        if target == "unseen":
            mask_unseen = np.in1d(label_true.ravel(), unseen).reshape(label_true.shape)
            mask = mask & mask_unseen

        elif target == "seen":
            seen = [x for x in range(n_class) if x not in unseen]
            mask_seen = np.in1d(label_true.ravel(), seen).reshape(label_true.shape)
            mask = mask & mask_seen

        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def _fast_hist_specific_class(self, label_true, label_pred, n_class, target_class):
        mask = (label_true >= 0) & (label_true < n_class)
        mask_class = np.in1d(label_true.ravel(), target_class).reshape(label_true.shape)
        mask = mask & mask_class
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def _hist_to_metrics(self, hist):
        if hist.sum() == 0:
            acc = 0.0
        else:
            acc = np.diag(hist).sum() / hist.sum()

        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, mean_iu, fwavacc

    def label_accuracy_score(self, label_trues, label_preds, by_class=False):
        """Returns accuracy score evaluation result.
          - overall accuracy
          - mean accuracy
          - mean IU
          - fwavacc
        """
        hist = np.zeros((self.num_class, self.num_class))

        if self.unseen_classes_idx:
            unseen_hist, seen_hist = (
                np.zeros((self.num_class, self.num_class)),
                np.zeros((self.num_class, self.num_class)),
            )

        if by_class:
            class_hist = []
            for class_idx in range(self.num_class):
                class_hist.append(np.zeros((self.num_class, self.num_class)))

        for lt, lp in zip(label_trues, label_preds):
            hist += self._fast_hist(
                lt.flatten(), lp.flatten(), self.num_class, target="all"
            )
            if self.unseen_classes_idx:
                seen_hist += self._fast_hist(
                    lt.flatten(),
                    lp.flatten(),
                    self.num_class,
                    target="seen",
                    unseen=self.unseen_classes_idx,
                )
                unseen_hist += self._fast_hist(
                    lt.flatten(),
                    lp.flatten(),
                    self.num_class,
                    target="unseen",
                    unseen=self.unseen_classes_idx,
                )

            if by_class:
                unique = np.unique(lt.flatten()).astype(np.int32)
                for class_idx in unique:
                    if class_idx != 255:
                        class_hist[class_idx] += self._fast_hist_specific_class(
                            lt.flatten(), lp.flatten(), self.num_class, class_idx
                        )

        metrics = self._hist_to_metrics(hist)
        if self.unseen_classes_idx:
            seen_metrics, unseen_metrics = (
                self._hist_to_metrics(seen_hist),
                self._hist_to_metrics(unseen_hist),
            )
            metrics = metrics, seen_metrics, unseen_metrics

        if by_class:
            class_metrics = []
            for class_idx in range(self.num_class):
                class_metrics.append(self._hist_to_metrics(class_hist[class_idx]))

        metrics = metrics
        if by_class:
            return metrics, class_metrics
        else:
            return metrics
