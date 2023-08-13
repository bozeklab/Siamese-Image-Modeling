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


def unpack_predictions(predictions: dict, model, device) -> OrderedDict:
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
            [k, v.permute(0, 2, 3, 1).contiguous().to(device)]
            for k, v in predictions.items()
            if k != "tissue_types"
        ]
    )
    predictions_["tissue_types"] = predictions["tissue_types"].to(device)
    predictions_["nuclei_binary_map"] = F.softmax(
        predictions_["nuclei_binary_map"], dim=-1
    )  # shape: (batch_size, H, W, 2)
    predictions_["nuclei_type_map"] = F.softmax(
        predictions_["nuclei_type_map"], dim=-1
    )  # shape: (batch_size, H, W, num_nuclei_classes)
    (
        predictions_["instance_map"],
        predictions_["instance_types"],
    ) = model.calculate_instance_map(
        predictions_)  # shape: (batch_size, H', W')
    predictions_["instance_types_nuclei"] = model.generate_instance_nuclei_map(
        predictions_["instance_map"], predictions_["instance_types"]
    ).to(
        device
    )  # shape: (batch_size, H, W, num_nuclei_classes)

    return predictions_


def train_unetr_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                          data_loader: Iterable, optimizer: torch.optim.Optimizer,
                          device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                          mixup_fn: Optional[Mixup] = None, log_writer=None,
                          args=None):
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
        x = x.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            predictions_ = model(x)
            predictions = unpack_predictions(predictions_, model, device)
            print('!!!!')
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
