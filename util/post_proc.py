import warnings
from typing import Tuple, Literal, OrderedDict, List

import cv2
import numpy as np
from scipy.ndimage import measurements
from scipy.ndimage.morphology import binary_fill_holes
from skimage.segmentation import watershed
import torch

from util.tools import get_bounding_box, remove_small_objects


class DetectionCellPostProcessor:
    def __init__(
        self,
        nr_types: int = None,
        magnification: Literal[20, 40] = 40,
        gt: bool = False,
    ) -> None:
        """DetectionCellPostProcessor for postprocessing prediction maps and get detected cells

        Args:
            nr_types (int, optional): Number of cell types, including background (background = 0). Defaults to None.
            magnification (Literal[20, 40], optional): Which magnification the data has. Defaults to 40.
            gt (bool, optional): If this is gt data (used that we do not suppress tiny cells that may be noise in a prediction map).
                Defaults to False.

        Raises:
            NotImplementedError: Unknown magnification
        """
        self.nr_types = nr_types
        self.magnification = magnification
        self.gt = gt

        if magnification == 40:
            self.object_size = 10
            self.k_size = 21
        elif magnification == 20:
            self.object_size = 5
            self.k_size = 13
        else:
            raise NotImplementedError("Unknown magnification")
        if gt:  # to not supress something in gt!
            self.object_size = 100
            self.k_size = 21

    def post_process_cell_segmentation(
        self,
        pred_map: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """Post processing of one image tile

        Args:
            pred_map (np.ndarray): Combined output of tp, np and hv branches, in the same order. Shape: (H, W, 4)

        Returns:
            Tuple[np.ndarray, dict]:
                np.ndarray: Instance map for one image. Each nuclei has own integer. Shape: (H, W)
                dict: Instance dictionary. Main Key is the nuclei instance number (int), with a dict as value.
                    For each instance, the dictionary contains the keys: bbox (bounding box), centroid (centroid coordinates),
                    contour, type_prob (probability), type (nuclei type)
        """
        if self.nr_types is not None:
            pred_type = pred_map[..., :1]
            pred_inst = pred_map[..., 1:]
            pred_type = pred_type.astype(np.int32)
        else:
            pred_inst = pred_map

        pred_inst = np.squeeze(pred_inst)
        pred_inst = self.__proc_np_hv(
            pred_inst, object_size=self.object_size, ksize=self.k_size
        )

        inst_id_list = np.unique(pred_inst)[1:]  # exlcude background
        inst_info_dict = {}
        for inst_id in inst_id_list:
            inst_map = pred_inst == inst_id
            rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
            inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
            inst_map = inst_map[
                inst_bbox[0][0] : inst_bbox[1][0], inst_bbox[0][1] : inst_bbox[1][1]
            ]
            inst_map = inst_map.astype(np.uint8)
            inst_moment = cv2.moments(inst_map)
            inst_contour = cv2.findContours(
                inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # * opencv protocol format may break
            inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
            # < 3 points don't make a contour, so skip, likely artifact too
            # as the contours obtained via approximation => too small or sthg
            if inst_contour.shape[0] < 3:
                continue
            if len(inst_contour.shape) != 2:
                continue  # ! check for trickery shape
            inst_centroid = [
                (inst_moment["m10"] / inst_moment["m00"]),
                (inst_moment["m01"] / inst_moment["m00"]),
            ]
            inst_centroid = np.array(inst_centroid)
            inst_contour[:, 0] += inst_bbox[0][1]  # X
            inst_contour[:, 1] += inst_bbox[0][0]  # Y
            inst_centroid[0] += inst_bbox[0][1]  # X
            inst_centroid[1] += inst_bbox[0][0]  # Y
            inst_info_dict[inst_id] = {  # inst_id should start at 1
                "bbox": inst_bbox,
                "centroid": inst_centroid,
                "contour": inst_contour,
                "type_prob": None,
                "type": None,
            }
        #### * Get class of each instance id, stored at index id-1 (inst_id = number of deteced nucleus)
        for inst_id in list(inst_info_dict.keys()):
            rmin, cmin, rmax, cmax = (inst_info_dict[inst_id]["bbox"]).flatten()
            inst_map_crop = pred_inst[rmin:rmax, cmin:cmax]
            inst_type_crop = pred_type[rmin:rmax, cmin:cmax]
            inst_map_crop = (
                inst_map_crop == inst_id
            )  # TODO: duplicated operation, may be expensive
            inst_type = inst_type_crop[inst_map_crop]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0:  # ! pick the 2nd most dominant if exist
                if len(type_list) > 1:
                    inst_type = type_list[1][0]
            type_dict = {v[0]: v[1] for v in type_list}
            type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)
            inst_info_dict[inst_id]["type"] = int(inst_type)
            inst_info_dict[inst_id]["type_prob"] = float(type_prob)

        return pred_inst, inst_info_dict

    def __proc_np_hv(
        self, pred: np.ndarray, object_size: int = 10, ksize: int = 21
    ) -> np.ndarray:
        """Process Nuclei Prediction with XY Coordinate Map and generate instance map (each instance has unique integer)

        Separate Instances (also overlapping ones) from binary nuclei map and hv map by using morphological operations and watershed

        Args:
            pred (np.ndarray): Prediction output, assuming. Shape: (H, W, 3)
                * channel 0 contain probability map of nuclei
                * channel 1 containing the regressed X-map
                * channel 2 containing the regressed Y-map
            object_size (int, optional): Smallest oject size for filtering. Defaults to 10
            k_size (int, optional): Sobel Kernel size. Defaults to 21
        Returns:
            np.ndarray: Instance map for one image. Each nuclei has own integer. Shape: (H, W)
        """
        pred = np.array(pred, dtype=np.float32)

        blb_raw = pred[..., 0]
        h_dir_raw = pred[..., 1]
        v_dir_raw = pred[..., 2]

        # processing
        blb = np.array(blb_raw >= 0.5, dtype=np.int32)

        blb = measurements.label(blb)[0]
        blb = remove_small_objects(blb, min_size=object_size)
        blb[blb > 0] = 1  # background is 0 already

        h_dir = cv2.normalize(
            h_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
        v_dir = cv2.normalize(
            v_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )

        sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize)
        sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize)

        sobelh = 1 - (
            cv2.normalize(
                sobelh,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
        )
        sobelv = 1 - (
            cv2.normalize(
                sobelv,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
        )

        overall = np.maximum(sobelh, sobelv)
        overall = overall - (1 - blb)
        overall[overall < 0] = 0

        dist = (1.0 - overall) * blb
        ## nuclei values form mountains so inverse to get basins
        dist = -cv2.GaussianBlur(dist, (3, 3), 0)

        overall = np.array(overall >= 0.4, dtype=np.int32)

        marker = blb - overall
        marker[marker < 0] = 0
        marker = binary_fill_holes(marker).astype("uint8")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
        marker = measurements.label(marker)[0]
        marker = remove_small_objects(marker, min_size=object_size)

        proced_pred = watershed(dist, markers=marker, mask=blb)

        return proced_pred


def calculate_instances(pred_types, pred_insts):
    """Best used for GT

    Args:
        pred_types (_type_): _description_
        pred_insts (_type_): _description_

    Returns:
        _type_: _description_
    """
    type_preds = []
    for i in range(pred_types.shape[0]):
        pred_type = torch.argmax(pred_types, dim=-1)[i].detach().cpu().numpy()
        pred_inst = pred_insts[i].detach().cpu().numpy()
        inst_id_list = np.unique(pred_inst)[1:]  # exlcude background
        inst_info_dict = {}
        for inst_id in inst_id_list:
            inst_map = pred_inst == inst_id
            # TODO: change format of bbox output
            rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
            inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
            inst_map = inst_map[
                inst_bbox[0][0] : inst_bbox[1][0], inst_bbox[0][1] : inst_bbox[1][1]
            ]
            inst_map = inst_map.astype(np.uint8)
            inst_moment = cv2.moments(inst_map)
            inst_contour = cv2.findContours(
                inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # * opencv protocol format may break
            inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
            # < 3 points dont make a contour, so skip, likely artifact too
            # as the contours obtained via approximation => too small or sthg
            if inst_contour.shape[0] < 3:
                continue
            if len(inst_contour.shape) != 2:
                continue  # ! check for trickery shape
            inst_centroid = [
                (inst_moment["m10"] / inst_moment["m00"]),
                (inst_moment["m01"] / inst_moment["m00"]),
            ]
            inst_centroid = np.array(inst_centroid)
            inst_contour[:, 0] += inst_bbox[0][1]  # X
            inst_contour[:, 1] += inst_bbox[0][0]  # Y
            inst_centroid[0] += inst_bbox[0][1]  # X
            inst_centroid[1] += inst_bbox[0][0]  # Y
            inst_info_dict[inst_id] = {  # inst_id should start at 1
                "bbox": inst_bbox,
                "centroid": inst_centroid,
                "contour": inst_contour,
                "type_prob": None,
                "type": None,
            }
        #### * Get class of each instance id, stored at index id-1 (inst_id = number of deteced nucleus)
        for inst_id in list(inst_info_dict.keys()):
            rmin, cmin, rmax, cmax = (inst_info_dict[inst_id]["bbox"]).flatten()
            inst_map_crop = pred_inst[rmin:rmax, cmin:cmax]
            inst_type_crop = pred_type[rmin:rmax, cmin:cmax]
            inst_map_crop = (
                inst_map_crop == inst_id
            )  # TODO: duplicated operation, may be expensive
            inst_type = inst_type_crop[inst_map_crop]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0:  # ! pick the 2nd most dominant if exist
                if len(type_list) > 1:
                    inst_type = type_list[1][0]
            type_dict = {v[0]: v[1] for v in type_list}
            type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)
            inst_info_dict[inst_id]["type"] = int(inst_type)
            inst_info_dict[inst_id]["type_prob"] = float(type_prob)
        type_preds.append(inst_info_dict)

    return type_preds


def calculate_instance_map(predictions: OrderedDict, num_nuclei_classes, magnification=40):
    """Calculate Instance Map from network predictions (after Softmax output)

    Args:
        predictions (dict): Dictionary with the following required keys:
            * nuclei_binary_map: Binary Nucleus Predictions. Shape: (batch_size, H, W, 2)
            * nuclei_type_map: Type prediction of nuclei. Shape: (batch_size, H, W, 6)
            * hv_map: Horizontal-Vertical nuclei mapping. Shape: (batch_size, H, W, 2)
        magnification (Literal[20, 40], optional): Which magnification the data has. Defaults to 40.

    Returns:
        Tuple[torch.Tensor, List[dict]]:
            * torch.Tensor: Instance map. Each Instance has own integer. Shape: (batch_size, H, W)
            * List of dictionaries. Each List entry is one image. Each dict contains another dict for each detected nucleus.
                For each nucleus, the following information are returned: "bbox", "centroid", "contour", "type_prob", "type"
    """
    cell_post_processor = DetectionCellPostProcessor(
        nr_types=num_nuclei_classes, magnification=magnification, gt=False
    )
    instance_preds = []
    type_preds = []
    for i in range(predictions["nuclei_binary_map"].shape[0]):
        pred_map = np.concatenate(
            [
                torch.argmax(predictions["nuclei_type_map"], dim=-1)[i].detach().cpu()[..., None],
                torch.argmax(predictions["nuclei_binary_map"], dim=-1)[i].detach().cpu()[..., None],
                predictions["hv_map"][i].detach().cpu(),
            ],
            axis=-1,
        )
        instance_pred = cell_post_processor.post_process_cell_segmentation(pred_map)
        instance_preds.append(instance_pred[0])
        type_preds.append(instance_pred[1])

    return torch.Tensor(np.stack(instance_preds)), type_preds


def generate_instance_nuclei_map(instance_maps: torch.Tensor, type_preds: List[dict], num_nuclei_classes):
    """Convert instance map (binary) to nuclei type instance map

    Args:
        instance_maps (torch.Tensor): Binary instance map, each instance has own integer. Shape: (batch_size, H, W)
        type_preds (List[dict]): List (len=batch_size) of dictionary with instance type information (compare post_process_hovernet function for more details)

    Returns:
        torch.Tensor: Nuclei type instance map. Shape: (batch_size, H, W, self.num_nuclei_classes)
    """
    batch_size, h, w = instance_maps.shape
    instance_type_nuclei_maps = torch.zeros(
        (batch_size, h, w, num_nuclei_classes)
    )
    for i in range(batch_size):
        instance_type_nuclei_map = torch.zeros((h, w, num_nuclei_classes))
        instance_map = instance_maps[i]
        type_pred = type_preds[i]
        for nuclei, spec in type_pred.items():
            nuclei_type = spec["type"]
            instance_type_nuclei_map[:, :, nuclei_type][
                instance_map == nuclei
            ] = nuclei

        instance_type_nuclei_maps[i, :, :, :] = instance_type_nuclei_map

    return instance_type_nuclei_maps