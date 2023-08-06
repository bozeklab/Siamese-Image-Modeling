import os
import random

import imageio
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from skimage.transform import resize
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from dataclasses import dataclass


@dataclass
class Config:
    input_path: str
    size: int


def random_file_selector(images_dir, labels_dir):
    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        raise ValueError("Invalid directory path. Please provide valid directory paths for both folders A and B.")

    imgs = [file for file in os.listdir(images_dir) if file.endswith(".png")]
    labels = [file for file in os.listdir(labels_dir) if file.endswith(".npy")]

    if not imgs or not labels:
        raise ValueError("No files found in one or both of the given directories.")

    selected_image = random.choice(imgs)
    corresponding_mask = selected_image.replace(".png", ".npy")
    return os.path.join(images_dir, selected_image), os.path.join(labels_dir, corresponding_mask)


args = Config(input_path='/Users/piotrwojcik/data/pannuke/fold1', size=352)

if __name__ == "__main__":
    img_pth, segmap_pth = random_file_selector(os.path.join(args.input_path, 'images'),
                                       os.path.join(args.input_path, 'labels'))

    img = imageio.v3.imread(img_pth)
    img = resize(img, (args.size, args.size))
    img = (img * 255.0).astype(np.uint8)
    segmap = np.load(segmap_pth, allow_pickle=True).item()['type_map']
    segmap = SegmentationMapsOnImage(segmap, shape=img.shape)
    segmap = segmap.resize(sizes=(args.size, args.size), interpolation="nearest")

    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
    ], random_order=True)

    # Augment images and segmaps.
    images_aug = []
    segmaps_aug = []
    for _ in range(5):
        images_aug_i, segmaps_aug_i = seq(image=img, segmentation_maps=segmap)
        images_aug.append(images_aug_i)
        segmaps_aug.append(segmaps_aug_i)

    # We want to generate an image containing the original input image and
    # segmentation maps before/after augmentation. (Both multiple times for
    # multiple augmentations.)
    #
    # The whole image is supposed to have five columns:
    # (1) original image,
    # (2) original image with segmap,
    # (3) augmented image,
    # (4) augmented segmap on augmented image,
    # (5) augmented segmap on its own in.
    #
    # We now generate the cells of these columns.
    #
    # Note that draw_on_image() and draw() both return lists of drawn
    # images. Assuming that the segmentation map array has shape (H,W,C),
    # the list contains C items.
    cells = []
    for image_aug, segmap_aug in zip(images_aug, segmaps_aug):
        cells.append(img)  # column 1
        cells.append(segmap.draw_on_image(img)[0])  # column 2
        cells.append(image_aug)  # column 3
        cells.append(segmap_aug.draw_on_image(image_aug)[0])  # column 4
        cells.append(segmap_aug.draw(size=image_aug.shape[:2])[0])  # column 5

    # Convert cells to a grid image and save.
    grid_image = ia.draw_grid(cells, cols=5)
    imageio.imwrite("example_segmaps_fr.jpg", grid_image)