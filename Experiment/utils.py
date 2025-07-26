import numpy as np
from sklearn.metrics import roc_auc_score, jaccard_score
import cv2
from torch import nn
import torch.nn.functional as F
import math
from functools import wraps
import warnings
import weakref
from torch.optim.optimizer import Optimizer
import torch


class WeightedBCE(nn.Module):
    def __init__(self, weights=[0.4, 0.6], n_labels=1):
        super(WeightedBCE, self).__init__()
        self.weights = weights
        self.n_labels = n_labels

    def forward(self, logit_pixel, truth_pixel):
        if self.n_labels == 1:
            logit = logit_pixel.view(-1)
            truth = truth_pixel.view(-1)

            truth = torch.clamp(truth, 0, 1)

            assert (logit.shape == truth.shape)
            loss = F.binary_cross_entropy(logit, truth, reduction='none')
            pos = (truth > 0.5).float()
            neg = (truth <= 0.5).float()

            pos_weight = pos.sum().item() + 1e-12
            neg_weight = neg.sum().item() + 1e-12
            loss = (self.weights[0] * pos * loss / pos_weight + self.weights[1] * neg * loss / neg_weight).sum()
            return loss


class WeightedDiceLoss(nn.Module):
    def __init__(self, weights=[0.5, 0.5], n_labels=1):
        super(WeightedDiceLoss, self).__init__()
        self.weights = weights
        self.n_labels = n_labels

    def forward(self, logit, truth, smooth=1e-5):
        if (self.n_labels == 1):
            batch_size = len(logit)
            logit = logit.view(batch_size, -1)
            truth = truth.view(batch_size, -1)

            truth = torch.clamp(truth, 0, 1)

            assert (logit.shape == truth.shape)
            p = logit.view(batch_size, -1)
            t = truth.view(batch_size, -1)
            w = truth.detach()
            w = w * (self.weights[1] - self.weights[0]) + self.weights[0]

            p = w * (p)
            t = w * (t)
            intersection = (p * t).sum(-1)
            union = (p * p).sum(-1) + (t * t).sum(-1)
            dice = 1 - (2 * intersection + smooth) / (union + smooth)

            loss = dice.mean()
            return loss


class WeightedDiceBCE(nn.Module):
    def __init__(self, dice_weight=1, BCE_weight=1, n_labels=1):
        super(WeightedDiceBCE, self).__init__()
        self.BCE_loss = WeightedBCE(weights=[0.5, 0.5], n_labels=n_labels)
        self.dice_loss = WeightedDiceLoss(weights=[0.5, 0.5], n_labels=n_labels)
        self.n_labels = n_labels
        self.BCE_weight = BCE_weight
        self.dice_weight = dice_weight

    def _show_dice(self, inputs, targets):
        if isinstance(inputs, tuple):
            inputs = inputs[0]

        inputs = inputs.clone()
        targets = targets.clone()

        inputs[inputs >= 0.5] = 1
        inputs[inputs < 0.5] = 0

        targets[targets > 0] = 1
        targets[targets <= 0] = 0
        hard_dice_coeff = 1.0 - self.dice_loss(inputs, targets)
        return hard_dice_coeff

    def forward(self, inputs, targets):
        targets = torch.clamp(targets, 0, 1)

        dice = self.dice_loss(inputs, targets)
        BCE = self.BCE_loss(inputs, targets)

        dice_BCE_loss = self.dice_weight * dice + self.BCE_weight * BCE

        return dice_BCE_loss


def auc_on_batch(masks, pred):
    aucs = []
    for i in range(pred.shape[1]):
        prediction = pred[i][0].cpu().detach().numpy()

        mask = masks[i].cpu().detach().numpy()

        mask = np.clip(mask, 0, 1)
        # print("rrr",np.max(mask), np.min(mask))
        aucs.append(roc_auc_score(mask.reshape(-1), prediction.reshape(-1)))
    return np.mean(aucs)


def iou_on_batch(mask, pred):
    if isinstance(pred, tuple):
        pred = pred[0]

    b = mask.shape[0]
    IoU = 0.0
    for i in range(pred.shape[0]):
        temp = pred[i].view(-1)
        temp = (temp > 0.5).float()

        m_temp = mask[i].view(-1)

        intersection = (temp * m_temp).sum()
        union = temp.sum() + m_temp.sum() - intersection

        IoU += (intersection + 1e-7) / (union + 1e-7)

    return IoU / b


def dice_coef(y_true, y_pred):
    smooth = 1e-5
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_on_batch(masks, pred):
    dices = []

    for i in range(pred.shape[0]):
        pred_tmp = pred[i][0].cpu().detach().numpy()
        mask_tmp = masks[i].cpu().detach().numpy()
        mask_tmp = np.clip(mask_tmp, 0, 1)
        pred_tmp[pred_tmp >= 0.5] = 1
        pred_tmp[pred_tmp < 0.5] = 0

        mask_tmp[mask_tmp > 0] = 1
        mask_tmp[mask_tmp <= 0] = 0
        dices.append(dice_coef(mask_tmp, pred_tmp))
    return np.mean(dices)


def save_on_batch(images1, masks, pred, names, vis_path):
    if isinstance(pred, tuple):
        pred = pred[0]

    for i in range(pred.shape[0]):
        pred_tmp = pred[i].cpu().detach().numpy()
        if pred_tmp.ndim == 3 and pred_tmp.shape[0] == 1:
            pred_tmp = pred_tmp.squeeze(0)
        mask_tmp = masks[i].cpu().detach().numpy()
        if mask_tmp.ndim == 3:
            mask_tmp = mask_tmp.squeeze()
        mask_tmp = np.clip(mask_tmp, 0, 1)
        pred_tmp = np.clip(pred_tmp, 0, 1)

        pred_tmp = (pred_tmp * 255).astype(np.uint8)
        mask_tmp = (mask_tmp * 255).astype(np.uint8)

        pred_tmp[pred_tmp >= 128] = 255
        pred_tmp[pred_tmp < 128] = 0
        mask_tmp[mask_tmp > 0] = 255
        mask_tmp[mask_tmp <= 0] = 0

        if pred_tmp.ndim > 2:
            pred_tmp = np.squeeze(pred_tmp)
        if mask_tmp.ndim > 2:
            mask_tmp = np.squeeze(mask_tmp)

        try:

            cv2.imwrite(vis_path + names[i][:-4] + "_pred.jpg", pred_tmp)
            cv2.imwrite(vis_path + names[i][:-4] + "_gt.jpg", mask_tmp)
        except Exception as e:
            print(f"Error saving image: {e}")
            print(f"pred_tmp shape: {pred_tmp.shape}, dtype: {pred_tmp.dtype}")
            print(f"mask_tmp shape: {mask_tmp.shape}, dtype: {mask_tmp.dtype}")
            pred_tmp_safe = np.array(pred_tmp, dtype=np.uint8)
            mask_tmp_safe = np.array(mask_tmp, dtype=np.uint8)
            if pred_tmp_safe.ndim == 2:
                cv2.imwrite(vis_path + names[i][:-4] + "_pred.jpg", pred_tmp_safe)
            if mask_tmp_safe.ndim == 2:
                cv2.imwrite(vis_path + names[i][:-4] + "_gt.jpg", mask_tmp_safe)


class _LRScheduler(object):

    def __init__(self, optimizer, last_epoch=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch

        def with_counter(method):
            if getattr(method, '_with_counter', False):
                return method

            instance_ref = weakref.ref(method.__self__)
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0

        self.step()

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):

        self.__dict__.update(state_dict)

    def get_last_lr(self):

        return self._last_lr

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = self._get_closed_form_lr()
                else:
                    values = self.get_lr()

        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class CosineAnnealingWarmRestarts(_LRScheduler):

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min

        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)

        self.T_cur = self.last_epoch

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", DeprecationWarning)

        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """Step could be called after every batch update

        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> iters = len(dataloader)
            >>> for epoch in range(20):
            >>>     for i, sample in enumerate(dataloader):
            >>>         inputs, labels = sample['inputs'], sample['labels']
            >>>         scheduler.step(epoch + i / iters)
            >>>         optimizer.zero_grad()
            >>>         outputs = net(inputs)
            >>>         loss = criterion(outputs, labels)
            >>>         loss.backward()
            >>>         optimizer.step()

        This function can be called in an interleaved way.

        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> for epoch in range(20):
            >>>     scheduler.step()
            >>> scheduler.step(26)
            >>> scheduler.step() # scheduler.step(27), instead of scheduler(20)
        """

        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]