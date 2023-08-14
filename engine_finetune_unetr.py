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
from sklearn.metrics import accuracy_score

from timm.data import Mixup
from timm.utils import accuracy
from collections import OrderedDict
import torch.nn.functional as F
from torchmetrics.functional import dice
from torchmetrics.functional.classification import binary_jaccard_index

import util.misc as misc
import util.lr_sched as lr_sched
import numpy as np
from models_unetr_vit import CellViT
from util.img_with_mask_dataset import PanNukeDataset
from util.metrics import remap_label, get_fast_pq
from util.post_proc import calculate_instance_map, generate_instance_nuclei_map


def unpack_predictions(predictions: dict, num_nuclei_classes, device) -> OrderedDict:
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
    ) = calculate_instance_map(predictions_, num_nuclei_classes)  # shape: (batch_size, H', W')
    predictions_["instance_types_nuclei"] = generate_instance_nuclei_map(
        predictions_["instance_map"], predictions_["instance_types"], num_nuclei_classes,
    ).to(
        device
    )  # shape: (batch_size, H, W, num_nuclei_classes)

    return predictions_


def unpack_masks(masks: dict, tissues_map, num_nuclei_classes, device) -> dict:
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


def calculate_loss(predictions: OrderedDict, gt: dict, loss_dict, device):
    """Calculate the loss

    Args:
        predictions (OrderedDict): OrderedDict: Processed network output. Keys are:
            * nuclei_binary_map: Softmax output for binary nuclei prediction branch. Shape: (batch_size, H, W, 2)
            * hv_map: Logit output for hv-prediction. Shape: (batch_size, H, W, 2)
            * nuclei_type_map: Softmax output for hv-prediction. Shape: (batch_size, H, W, 2)
            * tissue_types: Logit tissue prediction output. Shape: (batch_size, num_tissue_classes)
            * instance_map: Pixel-wise nuclear instance segmentation predictions. Shape: (batch_size, H, W)
            * instance_types: Dictionary, Pixel-wise nuclei type predictions
            * instance_types_nuclei: Pixel-wsie nuclear instance segmentation predictions, for each nuclei type. Shape: (batch_size, H, W, num_nuclei_classes)
        gt (dict): Ground truth values, with keys:
            * instance_map: Pixel-wise nuclear instance segmentations. Shape: (batch_size, H, W) -> each instance has one integer
            * nuclei_binary_map: One-Hot encoded binary map. Shape: (batch_size, H, W, 2)
            * hv_map: HV-map. Shape: (batch_size, H, W, 2)
            * nuclei_type_map: One-hot encoded nuclei type maps Shape: (batch_size, H, W, num_nuclei_classes)
            * instance_types_nuclei: Shape: (batch_size, H, W, num_nuclei_classes) -> instance has one integer, for each nuclei class
            * tissue_types: Tissue types, as torch.Tensor with integer values. Shape: batch_size

    Returns:
        torch.Tensor: Loss
    """
    total_loss = 0
    outputs = {}

    for branch, pred in predictions.items():
        if branch in [
            "instance_map",
            "instance_types",
            "instance_types_nuclei",
        ]:  # TODO: rather select branch from loss functions?
            continue
        branch_loss_fns = loss_dict[branch]
        for loss_name, loss_settings in branch_loss_fns.items():
            loss_fn = loss_settings["loss_fn"]
            weight = loss_settings["weight"]
            if loss_name == "msge":
                loss_value = loss_fn(
                    input=pred,
                    target=gt[branch],
                    focus=gt["nuclei_binary_map"][..., 1],
                    device=device,
                )
            else:
                loss_value = loss_fn(input=pred, target=gt[branch])
            total_loss = total_loss + weight * loss_value
            outputs[f"{branch}_{loss_name}"] = loss_value.item()

    return total_loss, outputs


def calculate_step_metric_train(predictions: dict, gt: dict) -> dict:
    """Calculate the metrics for the training step

    Args:
        predictions (OrderedDict): OrderedDict: Processed network output. Keys are:
            * nuclei_binary_map: Softmax output for binary nuclei prediction branch. Shape: (batch_size, H, W, 2)
            * hv_map: Logit output for hv-prediction. Shape: (batch_size, H, W, 2)
            * nuclei_type_map: Softmax output for hv-prediction. Shape: (batch_size, H, W, 2)
            * tissue_types: Logit tissue prediction output. Shape: (batch_size, num_tissue_classes)
            * instance_map: Pixel-wise nuclear instance segmentation predictions. Shape: (batch_size, H, W)
            * instance_types: Dictionary, Pixel-wise nuclei type predictions
            * instance_types_nuclei: Pixel-wsie nuclear instance segmentation predictions, for each nuclei type. Shape: (batch_size, H, W, num_nuclei_classes)
        gt (dict): Ground truth values, with keys:
            * instance_map: Pixel-wise nuclear instance segmentations. Shape: (batch_size, H, W) -> each instance has one integer
            * nuclei_binary_map: One-Hot encoded binary map. Shape: (batch_size, H, W, 2)
            * hv_map: HV-map. Shape: (batch_size, H, W, 2)
            * nuclei_type_map: One-hot encoded nuclei type maps Shape: (batch_size, H, W, num_nuclei_classes)
            * instance_types_nuclei: Shape: (batch_size, H, W, num_nuclei_classes) -> instance has one integer, for each nuclei class
            * tissue_types: Tissue types, as torch.Tensor with integer values. Shape: batch_size

    Returns:
        dict: Dictionary with metrics. Structure not fixed yet
    """
    # preparation and device movement
    predictions["tissue_types_classes"] = F.softmax(
        predictions["tissue_types"], dim=-1
    )
    pred_tissue = (
        torch.argmax(predictions["tissue_types_classes"], dim=-1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )
    predictions["instance_map"] = predictions["instance_map"].detach().cpu()
    predictions["instance_types_nuclei"] = (
        predictions["instance_types_nuclei"].detach().cpu().numpy().astype("int32")
    )
    gt["tissue_types"] = gt["tissue_types"].detach().cpu().numpy().astype(np.uint8)
    gt["nuclei_binary_map"] = torch.argmax(gt["nuclei_binary_map"], dim=-1).type(
        torch.uint8
    )
    gt["instance_types_nuclei"] = (
        gt["instance_types_nuclei"].detach().cpu().numpy().astype("int32")
    )

    tissue_detection_accuracy = accuracy_score(
        y_true=gt["tissue_types"], y_pred=pred_tissue
    )

    binary_dice_scores = []
    binary_jaccard_scores = []

    for i in range(len(pred_tissue)):
        # binary dice score: Score for cell detection per image, without background
        pred_binary_map = torch.argmax(predictions["nuclei_binary_map"][i], dim=-1)
        target_binary_map = gt["nuclei_binary_map"][i]
        cell_dice = (
            dice(preds=pred_binary_map, target=target_binary_map, ignore_index=0)
            .detach()
            .cpu()
        )
        binary_dice_scores.append(float(cell_dice))

        # binary aji
        cell_jaccard = (
            binary_jaccard_index(
                preds=pred_binary_map,
                target=target_binary_map,
            )
            .detach()
            .cpu()
        )
        binary_jaccard_scores.append(float(cell_jaccard))

    batch_metrics = {
        "tissue_detection_accuracy": tissue_detection_accuracy,
        "binary_dice_scores": binary_dice_scores,
        "binary_jaccard_scores": binary_jaccard_scores,
        "tissue_pred": pred_tissue,
        "tissue_gt": gt["tissue_types"],
    }

    return batch_metrics


def calculate_step_metric_validation(predictions: dict, gt: dict, num_classes):
    """Calculate the metrics for the validation step

    Args:
        predictions (OrderedDict): OrderedDict: Processed network output. Keys are:
            * nuclei_binary_map: Softmax output for binary nuclei prediction branch. Shape: (batch_size, H, W, 2)
            * hv_map: Logit output for hv-prediction. Shape: (batch_size, H, W, 2)
            * nuclei_type_map: Softmax output for hv-prediction. Shape: (batch_size, H, W, 2)
            * tissue_types: Logit tissue prediction output. Shape: (batch_size, num_tissue_classes)
            * instance_map: Pixel-wise nuclear instance segmentation predictions. Shape: (batch_size, H, W)
            * instance_types: Dictionary, Pixel-wise nuclei type predictions
            * instance_types_nuclei: Pixel-wsie nuclear instance segmentation predictions, for each nuclei type. Shape: (batch_size, H, W, num_nuclei_classes)
        gt (dict): Ground truth values, with keys:
            * instance_map: Pixel-wise nuclear instance segmentations. Shape: (batch_size, H, W) -> each instance has one integer
            * nuclei_binary_map: One-Hot encoded binary map. Shape: (batch_size, H, W, 2)
            * hv_map: HV-map. Shape: (batch_size, H, W, 2)
            * nuclei_type_map: One-hot encoded nuclei type maps Shape: (batch_size, H, W, num_nuclei_classes)
            * instance_types_nuclei: Shape: (batch_size, H, W, num_nuclei_classes) -> instance has one integer, for each nuclei class
            * tissue_types: Tissue types, as torch.Tensor with integer values. Shape: batch_size

    Returns:
        dict: Dictionary with metrics. Structure not fixed yet
    """
    # preparation and device movement
    predictions["tissue_types_classes"] = F.softmax(
        predictions["tissue_types"], dim=-1
    )
    pred_tissue = (
        torch.argmax(predictions["tissue_types_classes"], dim=-1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )
    predictions["instance_map"] = predictions["instance_map"].detach().cpu()
    predictions["instance_types_nuclei"] = (
        predictions["instance_types_nuclei"].detach().cpu().numpy().astype("int32")
    )
    instance_maps_gt = gt["instance_map"].detach().cpu()
    gt["tissue_types"] = gt["tissue_types"].detach().cpu().numpy().astype(np.uint8)
    gt["nuclei_binary_map"] = torch.argmax(gt["nuclei_binary_map"], dim=-1).type(
        torch.uint8
    )
    gt["instance_types_nuclei"] = (
        gt["instance_types_nuclei"].detach().cpu().numpy().astype("int32")
    )

    tissue_detection_accuracy = accuracy_score(
        y_true=gt["tissue_types"], y_pred=pred_tissue
    )

    binary_dice_scores = []
    binary_jaccard_scores = []
    cell_type_pq_scores = []
    pq_scores = []

    for i in range(len(pred_tissue)):
        # binary dice score: Score for cell detection per image, without background
        pred_binary_map = torch.argmax(predictions["nuclei_binary_map"][i], dim=-1)
        target_binary_map = gt["nuclei_binary_map"][i]
        cell_dice = (
            dice(preds=pred_binary_map, target=target_binary_map, ignore_index=0)
            .detach()
            .cpu()
        )
        binary_dice_scores.append(float(cell_dice))

        # binary aji
        cell_jaccard = (
            binary_jaccard_index(
                preds=pred_binary_map,
                target=target_binary_map,
            )
            .detach()
            .cpu()
        )
        binary_jaccard_scores.append(float(cell_jaccard))
        # pq values
        remapped_instance_pred = remap_label(predictions["instance_map"][i])
        remapped_gt = remap_label(instance_maps_gt[i])
        [_, _, pq], _ = get_fast_pq(true=remapped_gt, pred=remapped_instance_pred)
        pq_scores.append(pq)

        # pq values per class (skip background)
        nuclei_type_pq = []
        for j in range(0, num_classes):
            pred_nuclei_instance_class = remap_label(
                predictions["instance_types_nuclei"][i][..., j]
            )
            target_nuclei_instance_class = remap_label(
                gt["instance_types_nuclei"][i][..., j]
            )

            # if ground truth is empty, skip from calculation
            if len(np.unique(target_nuclei_instance_class)) == 1:
                pq_tmp = np.nan
            else:
                [_, _, pq_tmp], _ = get_fast_pq(
                    pred_nuclei_instance_class,
                    target_nuclei_instance_class,
                    match_iou=0.5,
                )
            nuclei_type_pq.append(pq_tmp)

        cell_type_pq_scores.append(nuclei_type_pq)

    batch_metrics = {
        "binary_dice_scores": binary_dice_scores,
        "binary_jaccard_scores": binary_jaccard_scores,
        "pq_scores": pq_scores,
        "cell_type_pq_scores": cell_type_pq_scores,
        "tissue_pred": pred_tissue,
        "tissue_gt": gt["tissue_types"],
    }

    return batch_metrics


def train_unetr_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                          device: torch.device, epoch: int, loss_scaler, num_nuclei_classes,
                          loss_setting, max_norm: float = 0, log_writer=None, args=None):
    model.train(True)

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    binary_dice_scores = []
    binary_jaccard_scores = []
    tissue_pred = []
    tissue_gt = []

    for data_iter_step, sample in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        x = sample['x']
        x = x.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            predictions_ = model(x)
            predictions = unpack_predictions(predictions_, num_nuclei_classes, device)
            gt = unpack_masks(masks=sample, device=device, tissues_map=PanNukeDataset.tissue_types,
                              num_nuclei_classes=num_nuclei_classes)

            loss, outputs = calculate_loss(predictions, gt, loss_setting, device)
            metric_logger.update(**outputs)

        loss_value = loss.item()
        batch_metrics = calculate_step_metric_train(predictions, gt)
        binary_dice_scores = (
                binary_dice_scores + batch_metrics["binary_dice_scores"]
        )
        binary_jaccard_scores = (
                binary_jaccard_scores + batch_metrics["binary_jaccard_scores"]
        )
        tissue_pred.append(batch_metrics["tissue_pred"])
        tissue_gt.append(batch_metrics["tissue_gt"])

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

        # calculate global metrics
    binary_dice_scores = np.array(binary_dice_scores)
    binary_jaccard_scores = np.array(binary_jaccard_scores)
    tissue_detection_accuracy = accuracy_score(y_true=np.concatenate(tissue_gt),
                                               y_pred=np.concatenate(tissue_pred))

    scalar_metrics = {
        "Binary-Cell-Dice-Mean/Train": np.nanmean(binary_dice_scores),
        "Binary-Cell-Jacard-Mean/Train": np.nanmean(binary_jaccard_scores),
        "Tissue-Multiclass-Accuracy/Train": tissue_detection_accuracy,
    }


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print("Scalar metrics for epoch [{}]".format(epoch))
    print("-----------------------")
    for key, value in scalar_metrics.items():
        print(f"{key}\t\t{value:.2f}")

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def unetr_evaluate(data_loader, model, num_nuclei_classes,
                   tissue_types, nuclei_types, reverse_tissue_types, device):

    # switch to evaluation mode
    model.eval()

    binary_dice_scores = []
    binary_jaccard_scores = []
    pq_scores = []
    cell_type_pq_scores = []
    tissue_pred = []
    tissue_gt = []

    for sample in data_loader:
        x = sample['x']
        x = x.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            predictions_ = model(x)
            predictions = unpack_predictions(predictions_, num_nuclei_classes, device)
            gt = unpack_masks(masks=sample, device=device, tissues_map=PanNukeDataset.tissue_types,
                              num_nuclei_classes=num_nuclei_classes)

        batch_metrics = calculate_step_metric_validation(predictions, gt, num_nuclei_classes)
        binary_dice_scores = (
                binary_dice_scores + batch_metrics["binary_dice_scores"]
        )
        binary_jaccard_scores = (
                binary_jaccard_scores + batch_metrics["binary_jaccard_scores"]
        )
        pq_scores = pq_scores + batch_metrics["pq_scores"]
        cell_type_pq_scores = (
                cell_type_pq_scores + batch_metrics["cell_type_pq_scores"]
        )
        tissue_pred.append(batch_metrics["tissue_pred"])
        tissue_gt.append(batch_metrics["tissue_gt"])

    tissue_types_val = [
        reverse_tissue_types[t].lower() for t in np.concatenate(tissue_gt)
    ]

    # calculate global metrics
    binary_dice_scores = np.array(binary_dice_scores)
    binary_jaccard_scores = np.array(binary_jaccard_scores)
    pq_scores = np.array(pq_scores)
    tissue_detection_accuracy = accuracy_score(
        y_true=np.concatenate(tissue_gt), y_pred=np.concatenate(tissue_pred)
    )

    scalar_metrics = {
        "Binary-Cell-Dice-Mean/Validation": np.nanmean(binary_dice_scores),
        "Binary-Cell-Jacard-Mean/Validation": np.nanmean(binary_jaccard_scores),
        "Tissue-Multiclass-Accuracy/Validation": tissue_detection_accuracy,
        "bPQ/Validation": np.nanmean(pq_scores),
        "mPQ/Validation": np.nanmean(
            [np.nanmean(pq) for pq in cell_type_pq_scores]
        ),
    }

    for tissue in tissue_types.keys():
        tissue = tissue.lower()
        tissue_ids = np.where(np.asarray(tissue_types_val) == tissue)
        scalar_metrics[f"{tissue}-Dice/Validation"] = np.nanmean(
            binary_dice_scores[tissue_ids]
        )
        scalar_metrics[f"{tissue}-Jaccard/Validation"] = np.nanmean(
            binary_jaccard_scores[tissue_ids]
        )
        scalar_metrics[f"{tissue}-bPQ/Validation"] = np.nanmean(
            pq_scores[tissue_ids]
        )
        scalar_metrics[f"{tissue}-mPQ/Validation"] = np.nanmean(
            [np.nanmean(pq) for pq in np.array(cell_type_pq_scores)[tissue_ids]]
        )
        # calculate nuclei metrics
    for nuc_name, nuc_type in nuclei_types.items():
        if nuc_name.lower() == "background":
            continue
        scalar_metrics[f"{nuc_name}-PQ/Validation"] = np.nanmean(
            [pq[nuc_type] for pq in cell_type_pq_scores]
        )


    # gather the stats from all processes
    print("Scalar validation metrics")
    print("-----------------------")
    for key, value in scalar_metrics.items():
        print(f"{key}\t\t\t{value:.2f}")

