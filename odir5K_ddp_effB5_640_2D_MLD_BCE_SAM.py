import os
import cv2
import yaml
import math
import timm
import time
import torch
import random
import logging
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from datetime import datetime
from pathlib import Path
import albumentations as A
import torch.optim as optim
import torch.nn.functional as F
from contextlib import suppress
from collections import OrderedDict
from sklearn import model_selection
import torch.utils.data as torchdata
from torch.optim.optimizer import Optimizer
from albumentations.pytorch import ToTensorV2
from timm.utils import AverageMeter, CheckpointSaver, NativeScaler
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import myCheckpointSaver as ckptSaver

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from timm.data import create_loader
from timm.models import convert_sync_batchnorm
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--pin-mem', action='store_true', default=True,help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)

    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def gather_predict_label(eval_metrics):

    world_size = dist.get_world_size()
    predictions_list = [torch.zeros_like(eval_metrics['predictions']) for _ in range(world_size)]
    validlabel_list = [torch.zeros_like(eval_metrics['valid_label']) for _ in range(world_size)]
    
    dist.all_gather(predictions_list, eval_metrics['predictions'])
    dist.all_gather(validlabel_list, eval_metrics['valid_label'])
    
    return torch.cat(predictions_list),torch.cat(validlabel_list)

class CFG:
    ######################
    # Globals #
    ######################
    seed = 55
    epochs = 40
    train = True
    oof = True
    inference = True
    folds = [0, 1, 2]
    img_size = 640
    main_metric = "epoch_score"
    minimize_metric = False
    distribute = False
    rank = -1
    world_size = -1

    ######################
    # Data #
    ######################
    basePath = "../../data/ODIR/"
    train_datadir = Path(basePath + "/Training Set/Images/")
    test_datadir_off = Path(basePath + "/Off-site Test Set/Images/")
    test_datadir_on = Path(basePath + "/On-site Test Set/Images/")
    train_csv = basePath + "/Training Set/Annotation/training annotation (English).xlsx"
    test_csv_off = basePath + "/Off-site Test Set/Annotation/off-site test annotation (English).xlsx"
    test_csv_on = basePath + "/On-site Test Set/Annotation/on-site test annotation (English).xlsx"

    ######################
    # Dataset #
    ######################
    target_columns = ["N", "D",	"G", "C", "A", "H", "M", "O"]

    ######################
    # Loaders #
    ######################
    loader_params = {
        "train": {
            "batch_size": 2,
            "num_workers": 4,
            "is_training": True
        },
        "valid": {
            "batch_size": 16,
            "num_workers": 4,
            "is_training": False
        },
        "test": {
            "batch_size": 8,
            "num_workers": 4,
            "is_training": False
        }
    }

    ######################
    # Split #
    ######################
    split = "MultilabelStratifiedKFold"
    split_params = {
        "n_splits": 3,
        "shuffle": True,
        "random_state": 1213
    }

    ######################
    # Model #
    ######################
    base_model_name = "tf_efficientnet_b5_ns"
    pooling = "GeM"
    pretrained = True
    num_classes = 8

    ######################
    # Criterion #
    ######################
    loss_name = "BCEFocalLoss"
    loss_params: dict = {}

    ######################
    # Optimizer #
    ######################
    optimizer_name = "SAM"
    base_optimizer = "Adam"
    optimizer_params = {
        "lr": 0.0005
    }
    # For SAM optimizer
    base_optimizer = "Adam"

    ######################
    # Scheduler #
    ######################
    scheduler_name = "CosineAnnealingLR"
    scheduler_params = {
        "T_max": 10
    }


def crop_image_from_gray(image: np.ndarray, threshold: int = 7):
    if image.ndim == 2:
        mask = image > threshold
        return image[np.ix_(mask.any(1), mask.any(0))]
    elif image.ndim == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray_image > threshold

    check_shape = image[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
    if (check_shape == 0):
        return image
    else:
        image1 = image[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        image2 = image[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        image3 = image[:, :, 2][np.ix_(mask.any(1), mask.any(0))]

        image = np.stack([image1, image2, image3], axis=-1)
        return image


class TrainDataset(torchdata.Dataset):
    def __init__(self, df: pd.DataFrame, datadir: Path, datadir_off: Path, datadir_on: Path, idList, target_columns: list, transform=None,
                 center_crop=True):
        self.df = df
        self.filenames = df["ID"].values
        self.datadir = datadir
        self.datadir_off = datadir_off
        self.datadir_on = datadir_on
        self.idList = idList
        self.target_columns = target_columns
        self.labels = df[target_columns].values
        self.transform = transform
        self.center_crop = center_crop

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int):
        filenameLeftt = str(self.filenames[index]) + "_left"
        filenameRight = str(self.filenames[index]) + "_right"
        filenameLefttNum = self.filenames[index]
        if filenameLefttNum in self.idList[0]:
            pathLeftt = self.datadir / f"{filenameLeftt}.jpg"
            pathRight = self.datadir / f"{filenameRight}.jpg"
        elif filenameLefttNum in self.idList[1]:
            pathLeftt = self.datadir_off / f"{filenameLeftt}.jpg"
            pathRight = self.datadir_off / f"{filenameRight}.jpg"
        else:
            pathLeftt = self.datadir_on / f"{filenameLeftt}.jpg"
            pathRight = self.datadir_on / f"{filenameRight}.jpg"

        imageLeftt = cv2.cvtColor(cv2.imread(str(pathLeftt)), cv2.COLOR_BGR2RGB)
        imageRight = cv2.cvtColor(cv2.imread(str(pathRight)), cv2.COLOR_BGR2RGB)
        if self.center_crop:
            imageLeftt = crop_image_from_gray(imageLeftt)
            imageRight = crop_image_from_gray(imageRight)

        if self.transform:
            augmented = self.transform(image = imageLeftt)
            imageLeftt = augmented["image"]

            augmented = self.transform(image = imageRight)
            imageRight = augmented["image"]

        label = torch.tensor(self.labels[index]).float()
        return {
            "imageLeftt": imageLeftt,
            "imageRight": imageRight,
            "targets": label
        }


class TestDataset(torchdata.Dataset):
    def __init__(self, df: pd.DataFrame, datadir: Path, transform=None, center_crop=True):
        self.df = df
        self.filenames = df["ID"].values
        self.datadir = datadir
        self.transform = transform
        self.center_crop = center_crop

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int):
        filenameLeftt = str(self.filenames[index]) + "_left"
        filenameRight = str(self.filenames[index]) + "_right"
        pathLeftt = self.datadir / f"{filenameLeftt}.jpg"
        pathRight = self.datadir / f"{filenameRight}.jpg"
        imageLeftt = cv2.cvtColor(cv2.imread(str(pathLeftt)), cv2.COLOR_BGR2RGB)
        imageRight = cv2.cvtColor(cv2.imread(str(pathRight)), cv2.COLOR_BGR2RGB)
        if self.center_crop:
            imageLeftt = crop_image_from_gray(imageLeftt)
            imageRight = crop_image_from_gray(imageRight)

        if self.transform:
            augmented = self.transform(image = imageLeftt)
            imageLeftt = augmented["image"]

            augmented = self.transform(image = imageRight)
            imageRight = augmented["image"]
        return {
            "imageLeftt": imageLeftt,
            "imageRight": imageRight
        }


# =================================================
# Transforms #
# =================================================
def get_transforms(img_size: int, mode="train"):
    if mode == "train":
        return A.Compose([
            A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.9, 1.1), ratio=(0.9, 1.1), p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=180, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=0.5),
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
            ToTensorV2()
        ])
    elif mode == "valid":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
            ToTensorV2()
        ])

def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def gem(x: torch.Tensor, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + f"(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

from srcModels.ml_decoder.ml_decoder import MLDecoder
class TimmModel(nn.Module):
    def __init__(self, base_model_name="tf_efficientnet_b0_ns", pooling="GeM", pretrained=True, num_classes=24):

        if False:
            super().__init__()
            self.right_model = timm.create_model(base_model_name, pretrained=pretrained, checkpoint_path="../preweights/tf_efficientnet_b5_ns-6f26d0cf.pth")
            if hasattr(self.right_model, "classifier"):
                in_features = self.right_model.classifier.in_features
                self.right_model.classifier = Identity()
                self.classifier = nn.Linear(in_features, num_classes)
            self.init_layer()
        else:
            super().__init__()
            self.right_model = timm.create_model(base_model_name, pretrained=pretrained, checkpoint_path="../preweights/tf_efficientnet_b5_ns-6f26d0cf.pth")
            if hasattr(self.right_model, "classifier"):
                in_features = self.right_model.classifier.in_features
                self.right_model.global_pool = Identity()
                self.right_model.classifier = Identity()
                self.classifier = MLDecoder(num_classes = CFG.num_classes, initial_num_features = in_features, num_of_groups = -1, decoder_embedding = 1536, zsl = 0)

    def init_layer(self):
        init_layer(self.classifier)

    def forward(self, left, right):
        output = self.right_model(torch.cat((left, right),3))
        return self.classifier(output)


# =================================================
# Criterion #
# =================================================
# https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075
class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        probas = torch.sigmoid(preds)
        loss = targets * self.alpha * (1. - probas)**self.gamma * bce_loss + (1. - targets) * probas**self.gamma * bce_loss
        loss = loss.mean()
        return loss


__CRITERIONS__ = {
    "BCEFocalLoss": BCEFocalLoss,
}


def get_criterion():
    if hasattr(nn, CFG.loss_name):
        return nn.__getattribute__(CFG.loss_name)(**CFG.loss_params)
    elif __CRITERIONS__.get(CFG.loss_name) is not None:
        return __CRITERIONS__[CFG.loss_name](**CFG.loss_params)
    else:
        raise NotImplementedError


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm


version_higher = (torch.__version__ >= "1.5.0")

class AdaBelief(Optimizer):
    r"""Implements AdaBelief algorithm. Modified from Adam in PyTorch
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        weight_decouple (boolean, optional): ( default: False) If set as True, then
            the optimizer uses decoupled weight decay as in AdamW
        fixed_decay (boolean, optional): (default: False) This is used when weight_decouple
            is set as True.
            When fixed_decay == True, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay$.
            When fixed_decay == False, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay \times lr$. Note that in this case, the
            weight decay ratio decreases with learning rate (lr).
        rectify (boolean, optional): (default: False) If set as True, then perform the rectified
            update similar to RAdam
    reference: AdaBelief Optimizer, adapting stepsizes by the belief in observed gradients
               NeurIPS 2020 Spotlight
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, weight_decouple=False, fixed_decay=False, rectify=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdaBelief, self).__init__(params, defaults)
        self.weight_decouple = weight_decouple
        self.rectify = rectify
        self.fixed_decay = fixed_decay
        if self.weight_decouple:
            print('Weight decoupling enabled in AdaBelief')
            if self.fixed_decay:
                print('Weight decay fixed')
        if self.rectify:
            print('Rectification enabled in AdaBelief')
        if amsgrad:
            print('AMS enabled in AdaBelief')

    def __setstate__(self, state):
        super(AdaBelief, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                amsgrad = group['amsgrad']
                # State initialization
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(
                    p.data,
                    memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state['exp_avg_var'] = torch.zeros_like(
                    p.data,
                    memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_var'] = torch.zeros_like(
                        p.data,
                        memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdaBelief does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                state = self.state[p]
                beta1, beta2 = group['betas']
                # State initialization
                if len(state) == 0:
                    state['rho_inf'] = 2.0 / (1.0 - beta2) - 1.0
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(
                        p.data,
                        memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_var'] = torch.zeros_like(
                        p.data,
                        memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_var'] = torch.zeros_like(
                            p.data,
                            memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)
                # get current state variable
                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                # perform weight decay, check if decoupled weight decay
                if self.weight_decouple:
                    if not self.fixed_decay:
                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        p.data.mul_(1.0 - group['weight_decay'])
                else:
                    if group['weight_decay'] != 0:
                        grad.add_(group['weight_decay'], p.data)
                # Update first and second moment running average
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(1 - beta2, grad_residual, grad_residual)
                if amsgrad:
                    max_exp_avg_var = state['max_exp_avg_var']
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_var, exp_avg_var, out=max_exp_avg_var)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_var.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_var.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                if not self.rectify:
                    # Default update
                    step_size = group['lr'] / bias_correction1
                    p.data.addcdiv_(-step_size, exp_avg, denom)
                else:  # Rectified update
                    # calculate rho_t
                    state['rho_t'] = state['rho_inf'] - 2 * state['step'] * beta2 ** state['step'] / (
                            1.0 - beta2 ** state['step'])
                    if state['rho_t'] > 4:  # perform Adam style update if variance is small
                        rho_inf, rho_t = state['rho_inf'], state['rho_t']
                        rt = (rho_t - 4.0) * (rho_t - 2.0) * rho_inf / (rho_inf - 4.0) / (rho_inf - 2.0) / rho_t
                        rt = math.sqrt(rt)
                        step_size = rt * group['lr'] / bias_correction1
                        p.data.addcdiv_(-step_size, exp_avg, denom)
                    else:  # perform SGD style update
                        p.data.add_(-group['lr'], exp_avg)
        return loss




__OPTIMIZERS__ = {
    "AdaBelief": AdaBelief,
    "SAM": SAM,
}


def get_optimizer(model: nn.Module, backbone, classifier):
    optimizer_name = CFG.optimizer_name
    if optimizer_name == "SAM":
        base_optimizer_name = CFG.base_optimizer
        if __OPTIMIZERS__.get(base_optimizer_name) is not None:
            base_optimizer = __OPTIMIZERS__[base_optimizer_name]
        else:
            base_optimizer = optim.__getattribute__(base_optimizer_name)
        return SAM([
                    {'params': backbone, 'lr': CFG.optimizer_params['lr']},
                    {'params': classifier, 'lr': CFG.optimizer_params['lr']}
                   ], base_optimizer)

    if __OPTIMIZERS__.get(optimizer_name) is not None:
        return __OPTIMIZERS__[optimizer_name](model.parameters(),
                                              **CFG.optimizer_params)
    else:
        return optim.__getattribute__(optimizer_name)([
                    {'params': backbone, 'lr': CFG.optimizer_params['lr']},
                    {'params': classifier, 'lr': CFG.optimizer_params['lr']}
        ])


def get_scheduler(optimizer):
    scheduler_name = CFG.scheduler_name

    if scheduler_name is None:
        return
    else:
        return optim.lr_scheduler.__getattribute__(scheduler_name)(
            optimizer, **CFG.scheduler_params)


# =================================================
# Split #
# =================================================
def get_split():
    if hasattr(model_selection, CFG.split):
        return model_selection.__getattribute__(CFG.split)(**CFG.scheduler_params)
    else:
        return MultilabelStratifiedKFold(**CFG.split_params)


# =================================================
# Utilities #
# =================================================
def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

from timm.utils import *
def train_epoch(epoch, model, loader, optimizer, loss_fn, loss_scaler, lr_scheduler=None, amp_autocast = suppress):

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    model.train()
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

    end = time.time()
    last_idx = len(loader) - 1
    for batch in enumerate(loader):

        batch_idx = batch[0]
        inputLeftt = batch[1]['imageLeftt']
        inputRight = batch[1]['imageRight']
        target = batch[1]['targets']
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        
        inputLeftt, inputRight, target = inputLeftt.cuda(), inputRight.cuda(), target.cuda()
        target = target.float()

        ## with amp_autocast():
        output = model(inputLeftt, inputRight)
        loss = loss_fn(output, target)

        loss.backward()
        optimizer.first_step(zero_grad = True)
        loss_fn(model(inputLeftt, inputRight), target).backward()
        optimizer.second_step(zero_grad = True)
        torch.cuda.synchronize()

        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % 48 == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            reduced_loss = reduce_tensor(loss.data, CFG.world_size)
            losses_m.update(reduced_loss.item(), inputLeftt.size(0))

            if args.local_rank == 0:
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=inputLeftt.size(0) / batch_time_m.val,
                        rate_avg=inputLeftt.size(0) / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))
        scheduler.step()
        end = time.time()
    
    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()
    return OrderedDict([('loss', losses_m.avg)])


def validate(model, loader, loss_fn, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()    
    preds = []  
    valid_label_ = []
    end = time.time()
    last_idx = len(loader) - 1
    model.eval()

    with torch.no_grad():
        for batch in enumerate(loader):

            batch_idx = batch[0]
            inputLeftt = batch[1]['imageLeftt']
            inputRight = batch[1]['imageRight']
            target = batch[1]['targets']

            last_batch = batch_idx == last_idx
            inputLeftt, inputRight = inputLeftt.cuda(), inputRight.cuda()
            target = target.cuda()
            target = target.float()

            output = model(inputLeftt, inputRight)
            loss = loss_fn(output, target)
            reduced_loss = reduce_tensor(loss.data, CFG.world_size)
            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), inputLeftt.size(0))
            preds.append(output.sigmoid().to('cpu').numpy())
            valid_label_.append(target.to('cpu').numpy())

            batch_time_m.update(time.time() - end)
            end = time.time()
            if (args.local_rank == 0 and last_batch or batch_idx % 48 == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m))

    predictions = np.concatenate(preds)
    predictions = torch.from_numpy(predictions)
    predictions = predictions.cuda()

    valid_label_ =  np.concatenate(valid_label_)
    valid_label_ = torch.from_numpy(valid_label_)
    valid_label_ = valid_label_.cuda()

    metrics = OrderedDict([('loss', losses_m.avg), ('predictions', predictions),('valid_label',valid_label_)])
    return metrics


class FormatterNoInfo(logging.Formatter):
    def __init__(self, fmt='%(levelname)s: %(message)s'):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):
        if record.levelno == logging.INFO:
            return str(record.getMessage())
        return logging.Formatter.format(self, record)


def setup_default_logging(default_level=logging.INFO, log_path=''):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(FormatterNoInfo())
    logging.root.addHandler(console_handler)
    logging.root.setLevel(default_level)
    if log_path:
        file_handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=(1024 ** 2 * 2), backupCount=3)
        file_formatter = logging.Formatter("%(asctime)s - %(name)20s: [%(levelname)8s] - %(message)s")
        file_handler.setFormatter(file_formatter)
        logging.root.addHandler(file_handler)


from sklearn.metrics import roc_auc_score, average_precision_score
def get_score(y_true, y_pred):
    scores = []
    mAPs = []
    for i in range(y_true.shape[1]):
        try: 
            score = roc_auc_score(y_true[:,i], y_pred[:,i])
            mAP = average_precision_score(y_true[:,i], y_pred[:,i], average="macro")
            mAPs.append(mAP)
            scores.append(score)
        except ValueError: 
            pass 
    return np.mean(scores), np.mean(mAPs)

from sklearn.metrics import f1_score
def json_map(cls_id, pred_json, ann_json):
    assert len(ann_json) == len(pred_json)
    predict = pred_json[:, cls_id]
    target = ann_json[:, cls_id]

    tmp = np.argsort(-predict)
    target = target[tmp]
    predict = predict[tmp]

    pre, obj = 0, 0
    for i in range(len(ann_json)):
        if target[i] == 1:
            obj += 1.0
            pre += obj / (i+1)
    pre /= obj
    return pre

def getSegmAP(label, predicted):
    apList = np.zeros(label.shape[1])
    for i in range(label.shape[1]):
        apList[i] = json_map(i, predicted, label)
    return np.mean(apList)

from torch.optim import lr_scheduler
from sklearn import metrics
def odir_metrics(gt_data, pr_data):
    th = 0.5
    gt = gt_data.flatten()
    pr = pr_data.flatten()
    kappa = metrics.cohen_kappa_score(gt, pr > th)
    f1 = metrics.f1_score(gt, pr > th, average='micro')
    auc = metrics.roc_auc_score(gt, pr)
    final_score = (kappa + f1 + auc) / 3.0
    return kappa, f1, auc, final_score


if __name__ == "__main__":

    _logger = logging.getLogger('train')
    args, args_text = _parse_args()
    # environment
    ## set_seed(CFG.seed)
    device = get_device()

    # validation
    splitter = get_split()

    # data
    train = pd.read_excel(CFG.train_csv)
    test_off = pd.read_excel(CFG.test_csv_off)
    test_on = pd.read_excel(CFG.test_csv_on)

    idList = []
    idList.append(train['ID'].values)
    idList.append(test_off['ID'].values)
    idList.append(test_on['ID'].values)

    catLabels = np.concatenate((train[CFG.target_columns].values , test_off[CFG.target_columns].values, test_on[CFG.target_columns].values))
    comLabels = pd.DataFrame()
    comLabels['ID'] = np.concatenate((train['ID'].values, test_off['ID'].values, test_on['ID'].values))
    comLabels[CFG.target_columns] = catLabels

    setup_default_logging()
    _logger.info('osivaz')

    amp_autocast = torch.cuda.amp.autocast
    loss_scaler = NativeScaler()

    if not os.path.isdir('ckpt_odir5K_ddp_effB5_640_2D_MLD_BCE_SAM/'):
        if args.local_rank == 0:
            os.mkdir('ckpt_odir5K_ddp_effB5_640_2D_MLD_BCE_SAM/')

    if 'WORLD_SIZE' in os.environ:
        CFG.distribute = int(os.environ['WORLD_SIZE']) > 1
        print(" Distribute  = ", CFG.distribute)

    for cv, (trn_idx, val_idx) in enumerate(splitter.split(comLabels, y=comLabels[CFG.target_columns])):

        device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        if cv == 0:
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
        CFG.world_size = torch.distributed.get_world_size()
        CFG.rank = torch.distributed.get_rank()
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.' % (CFG.rank, CFG.world_size))
        torch.manual_seed(CFG.seed + CFG.rank)

        pathFold = 'ckpt_odir5K_ddp_effB5_640_2D_MLD_BCE_SAM/CVSet' + str(cv)
        if args.local_rank == 0:
            os.mkdir(pathFold)

        model = TimmModel(base_model_name=CFG.base_model_name, pooling=CFG.pooling, 
                        pretrained=CFG.pretrained, num_classes=CFG.num_classes)
        if args.local_rank == 0:
            print("Model params = " + str(sum([m.numel() for m in model.parameters()])))
        
        model.cuda()
        model = convert_sync_batchnorm(model)
        model = NativeDDP(model, device_ids=[args.local_rank])  # can use device str in Torch >= 1.1

        trn_df = comLabels.loc[trn_idx, :].reset_index(drop=True)
        val_df = comLabels.loc[val_idx, :].reset_index(drop=True)
        dataset_train = TrainDataset(trn_df, CFG.train_datadir, CFG.test_datadir_off, CFG.test_datadir_on, idList, CFG.target_columns, transform=get_transforms(CFG.img_size, "train"))
        dataset_valid = TrainDataset(val_df, CFG.train_datadir, CFG.test_datadir_off, CFG.test_datadir_on, idList, CFG.target_columns, transform=get_transforms(CFG.img_size, "valid"))

        trainLoader = create_loader(dataset_train, input_size = CFG.img_size, transform = get_transforms(CFG.img_size, "train"), **CFG.loader_params['train'], distributed = True, 
                                    use_prefetcher = False, pin_memory = args.pin_mem)
        validLoader = create_loader(dataset_valid, input_size = CFG.img_size, transform = get_transforms(CFG.img_size, "valid"), **CFG.loader_params['valid'], distributed = True, 
                                    use_prefetcher = False, pin_memory = args.pin_mem)

        backbone, classifier = [], []
        for name, param in model.named_parameters():
            if 'classifier' in name:
                classifier.append(param)
            else:
                backbone.append(param)

        ## criterion = get_criterion()
        criterion = nn.BCEWithLogitsLoss().cuda()
        optimizer = get_optimizer(model, backbone, classifier)
        ## scheduler = get_scheduler(optimizer)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr = [0.0004, 0.0001], 
                    steps_per_epoch = len(trainLoader), epochs = CFG.epochs, pct_start = 0.2, cycle_momentum=False)

        eval_metric = 'EVAL_METRIC'
        decreasing = True if eval_metric == 'loss' else False
        saver = ckptSaver.CheckpointSaver(
            model = model, optimizer=optimizer, args = None, model_ema = None, amp_scaler=loss_scaler,
            checkpoint_dir=pathFold, recovery_dir=pathFold, decreasing=decreasing, max_history = 3)

        best_score = 0
        lastBestInd = -1
        for epoch in range(CFG.epochs):
            trainLoader.sampler.set_epoch(epoch)

            train_metrics = train_epoch(epoch + 1, model, trainLoader, optimizer, criterion, loss_scaler,
                lr_scheduler = scheduler, amp_autocast = amp_autocast)

            eval_metrics = validate(model, validLoader, criterion, amp_autocast=amp_autocast)
            predictions, valid_label = gather_predict_label(eval_metrics)
            valid_label = valid_label.detach().cpu().numpy()
            predictions = predictions.detach().cpu().numpy()
            AUC, mAP = get_score(valid_label, predictions)
            print("AUC score = %.4f" % AUC)
            print("mAP score = %.4f" % mAP)
            kappa, f1, auc, final_score = odir_metrics(valid_label, predictions)
            print("Kappa score:", kappa)
            print("F-1 score:", f1)
            print("AUC value:", auc)
            print("Final Score:", final_score)

            ## scheduler.step()
            if saver is not None:
                oldBestInd = lastBestInd
                lastBestInd = epoch
                best_score = final_score
                best_metric, best_epoch = saver.save_checkpoint(epoch + 1, metric = best_score)

        del model
        del optimizer
        torch.cuda.empty_cache()
