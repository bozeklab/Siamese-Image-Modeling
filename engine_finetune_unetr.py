# ------------------------------------------------------------------------
# SiameseIM
# Copyright (c) SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MAE (https://github.com/facebookresearch/mae)
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# References:
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# DeiT: https://github.com/facebookresearch/deit
# ------------------------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy
from collections import OrderedDict
import torch.nn.functional as F

import util.misc as misc
import util.lr_sched as lr_sched
from models_unetr_vit import CellViT
from util.img_with_mask_dataset import PanNukeDataset


def unpack_predictions(predictions: dict, num_nuclei_classes) -> OrderedDict:
    """Unpack the given predictions. Main focus lays on reshaping and postprocessing predictions, e.g. separating instances

    Args:
        predictions (dict): Dictionary with the following keys:
            * tissue_types: Logit tissue prediction output. Shape: (batch_size, num_tissue_classes)
            * nuclei_binary_map: Logit output for binary nuclei prediction branch. Shape: (batch_size, H, W, 2)
            * hv_map: Logit output for hv-prediction. Shape: (batch_size, H, W, 2)
            * nuclei_type_map: Logit output for nuclei instance-prediction. Shape: (batch_size, H, W, num_nuclei_classes)

    Returns:
        OrderedDict: Processed network output. Keys are:
            * nuclei_binary_map: Softmax output for binary nuclei prediction branch. Shape: (batch_size, H, W, 2)
            * hv_map: Logit output for hv-prediction. Shape: (batch_size, H, W, 2)
            * nuclei_type_map: Softmax output for hv-prediction. Shape: (batch_size, H, W, 2)
            * tissue_types: Logit tissue prediction output. Shape: (batch_size, num_tissue_classes)
            * instance_map: Pixel-wise nuclear instance segmentation predictions. Shape: (batch_size, H, W)
            * instance_types: Dictionary, Pixel-wise nuclei type predictions
            * instance_types_nuclei: Pixel-wsie nuclear instance segmentation predictions, for each nuclei type. Shape: (batch_size, H, W, num_nuclei_classes)
    """
    # assert
    # reshaping and postprocessing
    predictions_ = OrderedDict(
        [
            [k, v.permute(0, 2, 3, 1).contiguous()]
            for k, v in predictions.items()
            if k != "tissue_types"
        ]
    )
    predictions_["tissue_types"] = predictions["tissue_types"]
    predictions_["nuclei_binary_map"] = F.softmax(
        predictions_["nuclei_binary_map"], dim=-1
    )  # shape: (batch_size, H, W, 2)
    predictions_["nuclei_type_map"] = F.softmax(
        predictions_["nuclei_type_map"], dim=-1
    )  # shape: (batch_size, H, W, num_nuclei_classes)
    (
        predictions_["instance_map"],
        predictions_["instance_types"],
    ) = CellViT.calculate_instance_map(
        predictions_, num_nuclei_classes)  # shape: (batch_size, H', W')
    predictions_["instance_types_nuclei"] = CellViT.generate_instance_nuclei_map(
        predictions_["instance_map"], predictions_["instance_types"], num_nuclei_classes,
    ).to(
        predictions_["nuclei_binary_map"].device
    )  # shape: (batch_size, H, W, num_nuclei_classes)

    return predictions_


def unpack_masks(self, masks: dict, tissue_types: list,
                 tissues_map, num_nuclei_classes, device) -> dict:
    """Unpack the given masks. Main focus lays on reshaping and postprocessing masks to generate one dict

    Args:
        masks (dict): Required keys are:
            * instance_map: Pixel-wise nuclear instance segmentations. Shape: (batch_size, H, W)
            * nuclei_binary_map: Binary nuclei segmentations. Shape: (batch_size, H, W, 2)
            * hv_map: HV-Map. Shape: (batch_size, H, W, 2)
            * nuclei_type_map: Nuclei instance-prediction and segmentation (not binary, each instance has own integer). Shape: (batch_size, H, W, num_nuclei_classes)

        tissue_types (list): List of string names of ground-truth tissue types

    Returns:
        dict: Output ground truth values, with keys:
            * instance_map: Pixel-wise nuclear instance segmentations. Shape: (batch_size, H, W) -> each instance has one integer
            * nuclei_binary_map: One-Hot encoded binary map. Shape: (batch_size, H, W, 2)
            * hv_map: HV-map. Shape: (batch_size, H, W, 2)
            * nuclei_type_map: One-hot encoded nuclei type maps Shape: (batch_size, H, W, num_nuclei_classes)
            * instance_types_nuclei: Shape: (batch_size, H, W, num_nuclei_classes) -> instance has one integer, for each nuclei class
            * tissue_types: Tissue types, as torch.Tensor with integer values. Shape: batch_size
    """
    # get ground truth values, perform one hot encoding for segmentation maps
    gt_nuclei_binary_map_onehot = (
        F.one_hot(masks["nuclei_binary_map"], num_classes=2)
    ).type(
        torch.float32
    )  # background, nuclei
    nuclei_type_maps = torch.squeeze(masks["nuclei_type_map"]).type(torch.int64)
    gt_nuclei_type_maps_onehot = F.one_hot(
        nuclei_type_maps, num_classes=num_nuclei_classes
    ).type(
        torch.float32
    )  # background + nuclei types

    # assemble ground truth dictionary
    gt = {
        "nuclei_type_map": gt_nuclei_type_maps_onehot.to(
            device
        ),  # shape: (batch_size, H, W, num_nuclei_classes)
        "nuclei_binary_map": gt_nuclei_binary_map_onehot.to(
            device
        ),  # shape: (batch_size, H, W, 2)
        "hv_map": masks["hv_map"].to(device),  # shape: (batch_size, H, W, 2)
        "instance_map": masks["instance_map"].to(
            device
        ),  # shape: (batch_size, H, W) -> each instance has one integer
        "instance_types_nuclei": (
            gt_nuclei_type_maps_onehot * masks["instance_map"][..., None]
        ).to(
            device
        ),  # shape: (batch_size, H, W, num_nuclei_classes) -> instance has one integer, for each nuclei class
        "tissue_types": torch.Tensor([tissues_map[t] for t in masks["tissue_type"]])
        .type(torch.LongTensor)
        .to(device),  # shape: batch_size
    }
    return gt


def train_unetr_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                          data_loader: Iterable, optimizer: torch.optim.Optimizer,
                          device: torch.device, epoch: int, loss_scaler, num_nuclei_classes,
                          max_norm: float = 0, log_writer=None, args=None):
    model.train(True)

    binary_dice_scores = []
    binary_jaccard_scores = []
    tissue_pred = []
    tissue_gt = []
    train_example_img = None

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, sample in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        x = sample['x']

        print(sample['tissue_type'])

        x = x.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            predictions_ = model(x)
            predictions = unpack_predictions(predictions_, num_nuclei_classes)
            gt = unpack_masks(masks=sample, device=device, tissues_map=PanNukeDataset.tissue_types,
                              num_nuclei_classes=num_nuclei_classes)

            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
