import os

import torch
import torchvision.datasets as datasets
import imageio
import csv
import pickle
from scipy.ndimage import measurements
import numpy as np
import csv

from util.tools import get_bounding_box


class ImagesWithSegmentationMaps(datasets.VisionDataset):
    def __init__(self, root, transform):
        super(ImagesWithSegmentationMaps, self).__init__(root, transform=transform)

        self.root_imgs = os.path.join(root, 'images')
        self.root_masks = os.path.join(root, 'labels')
        self.transform = transform

        self.file_list = [filename for filename in os.listdir(self.root_imgs) if filename.endswith('.png')]

    def __getitem__(self, index):
        filename = self.file_list[index]
        image_path = os.path.join(self.root_imgs, filename)
        mask_path = os.path.join(self.root_masks, os.path.splitext(filename)[0] + '.npy')

        image = imageio.v3.imread(image_path)
        seg = np.load(mask_path, allow_pickle=True).item()
        type_map = seg['type_map']
        inst_map = seg['inst_map']

        return self.transform(image, type_map, inst_map)

    def __len__(self):
        return len(self.file_list)

    @staticmethod
    def gen_instance_hv_map(inst_map: np.ndarray) -> np.ndarray:
        """Obtain the horizontal and vertical distance maps for each
        nuclear instance.

        Args:
            inst_map (np.ndarray): Instance map with each instance labelled as a unique integer
                Shape: (H, W)
        Returns:
            np.ndarray: Horizontal and vertical instance map.
                Shape: (H, W, 2). First dimension is horizontal (horizontal gradient (-1 to 1)),
                last is vertical (vertical gradient (-1 to 1))
        """
        orig_inst_map = inst_map.copy()  # instance ID map

        x_map = np.zeros(orig_inst_map.shape[:2], dtype=np.float32)
        y_map = np.zeros(orig_inst_map.shape[:2], dtype=np.float32)

        inst_list = list(np.unique(orig_inst_map))
        inst_list.remove(0)  # 0 is background
        for inst_id in inst_list:
            inst_map = np.array(orig_inst_map == inst_id, np.uint8)
            inst_box = get_bounding_box(inst_map)

            # expand the box by 2px
            # Because we first pad the ann at line 207, the bboxes
            # will remain valid after expansion
            if inst_box[0] >= 2:
                inst_box[0] -= 2
            if inst_box[2] >= 2:
                inst_box[2] -= 2
            if inst_box[1] <= orig_inst_map.shape[0] - 2:
                inst_box[1] += 2
            if inst_box[3] <= orig_inst_map.shape[0] - 2:
                inst_box[3] += 2

            # improvement
            inst_map = inst_map[inst_box[0]: inst_box[1], inst_box[2]: inst_box[3]]

            if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
                continue

            # instance center of mass, rounded to nearest pixel
            inst_com = list(measurements.center_of_mass(inst_map))

            inst_com[0] = int(inst_com[0] + 0.5)
            inst_com[1] = int(inst_com[1] + 0.5)

            inst_x_range = np.arange(1, inst_map.shape[1] + 1)
            inst_y_range = np.arange(1, inst_map.shape[0] + 1)
            # shifting center of pixels grid to instance center of mass
            inst_x_range -= inst_com[1]
            inst_y_range -= inst_com[0]

            inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

            # remove coord outside of instance
            inst_x[inst_map == 0] = 0
            inst_y[inst_map == 0] = 0
            inst_x = inst_x.astype("float32")
            inst_y = inst_y.astype("float32")

            # normalize min into -1 scale
            if np.min(inst_x) < 0:
                inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
            if np.min(inst_y) < 0:
                inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
            # normalize max into +1 scale
            if np.max(inst_x) > 0:
                inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
            if np.max(inst_y) > 0:
                inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

            ####
            x_map_box = x_map[inst_box[0]: inst_box[1], inst_box[2]: inst_box[3]]
            x_map_box[inst_map > 0] = inst_x[inst_map > 0]

            y_map_box = y_map[inst_box[0]: inst_box[1], inst_box[2]: inst_box[3]]
            y_map_box[inst_map > 0] = inst_y[inst_map > 0]

        hv_map = np.dstack([x_map, y_map])
        return hv_map


class PanNukeDataset(ImagesWithSegmentationMaps):
    tissue_types = {"Adrenal_gland": 0, "Bile-duct": 1, "Bladder": 2, "Breast": 3, "Cervix": 4,
                    "Colon": 5, "Esophagus": 6, "HeadNeck": 7, "Kidney": 8, "Liver": 9, "Lung": 10, "Ovarian": 11,
                    "Pancreatic": 12, "Prostate": 13, "Skin": 14, "Stomach": 15, "Testis": 16, "Thyroid": 17,
                    "Uterus": 18}

    nuclei_types = {"Background": 0, "Neoplastic": 1, "Inflammatory": 2, "Connective": 3, "Dead": 4, "Epithelial": 5}

    def __len__(self):
        return len(self.file_list)

    def __init__(self, root, transform):
        self.tissue_dict = {}

        super(PanNukeDataset, self).__init__(root, transform=transform)

        with open(os.path.join(root, 'types.csv'), 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                img_name = row['img']
                tissue_type = row['type']
                self.tissue_dict[img_name] = tissue_type

    def __getitem__(self, index):
        filename = self.file_list[index]
        image_path = os.path.join(self.root_imgs, filename)
        mask_path = os.path.join(self.root_masks, os.path.splitext(filename)[0] + '.npy')

        image = imageio.v3.imread(image_path)
        seg = np.load(mask_path, allow_pickle=True).item()
        type_map = seg['type_map']
        inst_map = seg['inst_map']

        sample = self.transform(image, type_map, inst_map)
        sample['tissue_type'] = self.tissue_dict[filename]

        return sample




