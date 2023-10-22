import cv2
import numpy as np
import torch

from util.metrics import get_fast_pq
from util.pannuke_datasets import remap_label
from typing import Tuple


import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from torchmetrics.functional import dice
from torchmetrics.functional.classification import binary_jaccard_index
from scipy.optimize import linear_sum_assignment
import torch.nn as nn

def visualize_attention(attentions, w_featmap, h_featmap, patch_size=16, threshold=0.6):
    bsz, nh, num_patches, _ = attentions.size()

    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]
    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
    # interpolate
    th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[
        0].cpu().numpy()

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[
        0].cpu().numpy()
    return attentions


def attention_map_to_heatmap(attention_map, cmap='hot'):
    """
    Convert an attention map to an RGB heatmap.

    Args:
        attention_map (numpy.ndarray): A 2D attention map.
        cmap (str): Colormap name (e.g., 'hot', 'jet', 'viridis', 'inferno').

    Returns:
        heatmap (numpy.ndarray): An RGB heatmap.
    """
    # Normalize the attention map values to be in the range [0, 1].
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

    # Get the colormap.
    colormap = plt.get_cmap(cmap)

    # Apply the colormap to the attention map.
    heatmap = (colormap(attention_map)[:, :, :3] * 255).astype(np.uint8)
    return torch.tensor(heatmap).permute(2, 0, 1)


def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def pair_coordinates(
    setA: np.ndarray, setB: np.ndarray, radius: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use the Munkres or Kuhn-Munkres algorithm to find the most optimal
    unique pairing (largest possible match) when pairing points in set B
    against points in set A, using distance as cost function.

    Args:
        setA (np.ndarray): np.array (float32) of size Nx2 contains the of XY coordinate
                    of N different points
        setB (np.ndarray): np.array (float32) of size Nx2 contains the of XY coordinate
                    of N different points
        radius (float): valid area around a point in setA to consider
                a given coordinate in setB a candidate for match

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            pairing: pairing is an array of indices
                where point at index pairing[0] in set A paired with point
                in set B at index pairing[1]
            unparedA: remaining point in set A unpaired
            unparedB: remaining point in set B unpaired
    """
    # * Euclidean distance as the cost matrix
    pair_distance = scipy.spatial.distance.cdist(setA, setB, metric="euclidean")

    # * Munkres pairing with scipy library
    # the algorithm return (row indices, matched column indices)
    # if there is multiple same cost in a row, index of first occurence
    # is return, thus the unique pairing is ensured
    indicesA, paired_indicesB = linear_sum_assignment(pair_distance)

    # extract the paired cost and remove instances
    # outside of designated radius
    pair_cost = pair_distance[indicesA, paired_indicesB]

    pairedA = indicesA[pair_cost <= radius]
    pairedB = paired_indicesB[pair_cost <= radius]

    pairing = np.concatenate([pairedA[:, None], pairedB[:, None]], axis=-1)
    unpairedA = np.delete(np.arange(setA.shape[0]), pairedA)
    unpairedB = np.delete(np.arange(setB.shape[0]), pairedB)

    return pairing, unpairedA, unpairedB


def calculate_step_metric(predictions, gt, num_nuclei_classes, magnification=40):
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
        Tuple[dict, list]:
            * dict: Dictionary with metrics. Structure not fixed yet
            * list with cell_dice, cell_jaccard and pq for each image
    """

    # preparation and device movement
    predictions["tissue_types_classes"] = F.softmax(
        predictions["tissue_types"], dim=-1
    )
    pred_tissue = (
        torch.argmax(predictions["tissue_types_classes"], dim=-1).detach().cpu().numpy().astype(np.uint8)
    )
    predictions["instance_map"] = predictions["instance_map"]
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

    # segmentation scores
    binary_dice_scores = []  # binary dice scores per image
    binary_jaccard_scores = []  # binary jaccard scores per image
    pq_scores = []  # pq-scores per image
    dq_scores = []  # dq-scores per image
    sq_scores = []  # sq_scores per image
    cell_type_pq_scores = []  # pq-scores per cell type and image
    cell_type_dq_scores = []  # dq-scores per cell type and image
    cell_type_sq_scores = []  # sq-scores per cell type and image
    scores = []  # all scores in one list

    # detection scores
    paired_all = []  # unique matched index pair
    unpaired_true_all = (
        []
    )  # the index must exist in `true_inst_type_all` and unique
    unpaired_pred_all = (
        []
    )  # the index must exist in `pred_inst_type_all` and unique
    true_inst_type_all = []  # each index is 1 independent data point
    pred_inst_type_all = []  # each index is 1 independent data point

    # for detections scores
    true_idx_offset = 0
    pred_idx_offset = 0

    for i in range(len(pred_tissue)):
        # binary dice score: Score for cell detection per image, without background
        pred_binary_map = torch.argmax(predictions["nuclei_binary_map"][i], dim=-1)
        target_binary_map = gt["nuclei_binary_map"][i]
        cell_dice = (
            dice(preds=pred_binary_map, target=target_binary_map, ignore_index=0).detach().cpu()
        )
        binary_dice_scores.append(float(cell_dice))

        # binary aji
        cell_jaccard = (
            binary_jaccard_index(
                preds=pred_binary_map,
                target=target_binary_map,
            ).detach().cpu()
        )
        binary_jaccard_scores.append(float(cell_jaccard))

        # pq values
        remapped_instance_pred = remap_label(predictions["instance_map"][i])
        remapped_gt = remap_label(instance_maps_gt[i])
        [dq, sq, pq], _ = get_fast_pq(true=remapped_gt, pred=remapped_instance_pred)
        pq_scores.append(pq)
        dq_scores.append(dq)
        sq_scores.append(sq)
        scores.append(
            [
                cell_dice.detach().cpu().numpy(),
                cell_jaccard.detach().cpu().numpy(),
                pq,
            ]
        )

        # pq values per class (with class 0 beeing background -> should be skipped in the future)
        nuclei_type_pq = []
        nuclei_type_dq = []
        nuclei_type_sq = []
        for j in range(0, num_nuclei_classes):
            pred_nuclei_instance_class = remap_label(
                predictions["instance_types_nuclei"][i][..., j]
            )
            target_nuclei_instance_class = remap_label(
                gt["instance_types_nuclei"][i][..., j]
            )

            # if ground truth is empty, skip from calculation
            if len(np.unique(target_nuclei_instance_class)) == 1:
                pq_tmp = np.nan
                dq_tmp = np.nan
                sq_tmp = np.nan
            else:
                [dq_tmp, sq_tmp, pq_tmp], _ = get_fast_pq(
                    pred_nuclei_instance_class,
                    target_nuclei_instance_class,
                    match_iou=0.5,
                )
            nuclei_type_pq.append(pq_tmp)
            nuclei_type_dq.append(dq_tmp)
            nuclei_type_sq.append(sq_tmp)

        # detection scores
        true_centroids = np.array(
            [v["centroid"] for k, v in gt["instance_types"][i].items()]
        )
        true_instance_type = np.array(
            [v["type"] for k, v in gt["instance_types"][i].items()]
        )
        pred_centroids = np.array(
            [v["centroid"] for k, v in predictions["instance_types"][i].items()]
        )
        pred_instance_type = np.array(
            [v["type"] for k, v in predictions["instance_types"][i].items()]
        )

        if true_centroids.shape[0] == 0:
            true_centroids = np.array([[0, 0]])
            true_instance_type = np.array([0])
        if pred_centroids.shape[0] == 0:
            pred_centroids = np.array([[0, 0]])
            pred_instance_type = np.array([0])
        if magnification == 40:
            pairing_radius = 12
        else:
            pairing_radius = 6
        paired, unpaired_true, unpaired_pred = pair_coordinates(
            true_centroids, pred_centroids, pairing_radius
        )
        true_idx_offset = (
            true_idx_offset + true_inst_type_all[-1].shape[0] if i != 0 else 0
        )
        pred_idx_offset = (
            pred_idx_offset + pred_inst_type_all[-1].shape[0] if i != 0 else 0
        )
        true_inst_type_all.append(true_instance_type)
        pred_inst_type_all.append(pred_instance_type)

        # increment the pairing index statistic
        if paired.shape[0] != 0:  # ! sanity
            paired[:, 0] += true_idx_offset
            paired[:, 1] += pred_idx_offset
            paired_all.append(paired)

        unpaired_true += true_idx_offset
        unpaired_pred += pred_idx_offset
        unpaired_true_all.append(unpaired_true)
        unpaired_pred_all.append(unpaired_pred)

        cell_type_pq_scores.append(nuclei_type_pq)
        cell_type_dq_scores.append(nuclei_type_dq)
        cell_type_sq_scores.append(nuclei_type_sq)

    paired_all = np.concatenate(paired_all, axis=0)
    unpaired_true_all = np.concatenate(unpaired_true_all, axis=0)
    unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0)
    true_inst_type_all = np.concatenate(true_inst_type_all, axis=0)
    pred_inst_type_all = np.concatenate(pred_inst_type_all, axis=0)

    batch_metrics = {
        "binary_dice_scores": binary_dice_scores,
        "binary_jaccard_scores": binary_jaccard_scores,
        "pq_scores": pq_scores,
        "dq_scores": dq_scores,
        "sq_scores": sq_scores,
        "cell_type_pq_scores": cell_type_pq_scores,
        "cell_type_dq_scores": cell_type_dq_scores,
        "cell_type_sq_scores": cell_type_sq_scores,
        "tissue_pred": pred_tissue,
        "tissue_gt": gt["tissue_types"],
        "paired_all": paired_all,
        "unpaired_true_all": unpaired_true_all,
        "unpaired_pred_all": unpaired_pred_all,
        "true_inst_type_all": true_inst_type_all,
        "pred_inst_type_all": pred_inst_type_all,
    }

    return batch_metrics, scores


def remove_small_objects(pred, min_size=64, connectivity=1):
    """Remove connected components smaller than the specified size.

    This function is taken from skimage.morphology.remove_small_objects, but the warning
    is removed when a single label is provided.

    Args:
        pred: input labelled array
        min_size: minimum size of instance in output array
        connectivity: The connectivity defining the neighborhood of a pixel.

    Returns:
        out: output array with instances removed under min_size

    """
    out = pred

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndimage.generate_binary_structure(pred.ndim, connectivity)
        ccs = np.zeros_like(pred, dtype=np.int32)
        ndimage.label(pred, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out