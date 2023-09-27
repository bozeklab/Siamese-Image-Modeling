import argparse
import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL.Image import Image

from torch.utils.data import RandomSampler, BatchSampler, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import draw_bounding_boxes

from main_pretrain import DataAugmentationForSIMTraining, DataAugmentationForSIMFinetune
from util.datasets import ImagenetWithMask
from util.img_with_pickle_dataset import ImgWithPickledBoxesAndClassesDataset


def add_border(image):
    # Get the dimensions of the image tensor
    height, width, channels = image.shape

    # Create a new image with the border
    bordered_image = np.ones((height + 10, width + 10, channels), dtype=np.uint8) * 255

    # Insert the original image into the bordered image
    bordered_image[5:height + 5, 5:width + 5, :] = image

    return bordered_image


def gray_out_square(image, x_start, y_start, size, alpha):
    # Get the dimensions of the image tensor
    height, width, _ = image.shape

    # Calculate the end coordinates of the square region
    x_end = min(x_start + size, width)
    y_end = min(y_start + size, height)

    # Create a gray overlay image
    gray_overlay = alpha * image[y_start:y_end, x_start:x_end]

    # Replace the square region with the gray overlay
    image[y_start:y_end, x_start:x_end] = gray_overlay

    return image


def gray_out_mask(image, mask, patch_size, alpha):
    mh, mw = mask.shape

    for i in range(mh):
        for j in range(mw):
            if mask[i][j]:
                image = gray_out_square(image, i * patch_size, j * patch_size, patch_size, alpha)
    return image


def create_image_grid(images, num_cols=2):
    # Determine the dimensions of each image in the grid
    rows, cols, _ = images[0].shape

    # Determine the number of images and columns in the grid
    num_images = len(images)

    # Set the border size and color
    border_size = 5

    # Create a blank grid image to hold the combined grid
    grid_height = (rows + 2 * border_size) * (num_images // num_cols)
    grid_width = (cols + 2 * border_size) * num_cols
    grid = np.full((grid_height, grid_width, 3), 255)

    # Convert images to cv2 format with integer pixel values
    images = [np.uint8(image * 255) for image in images]

    # Populate the grid with the individual images and add borders
    for i, image in enumerate(images):
        row = i // num_cols
        col = i % num_cols
        x = col * cols
        y = row * rows

        # Add the image with border to the grid
        grid[y:y + rows + 2 * border_size, x:x + cols + 2 * border_size, :] = add_border(image)

    grid = grid.astype(np.uint8)

    scale_percent = 150  # percent of original size
    width = int(grid.shape[1] * scale_percent / 100)
    height = int(grid.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    grid = cv2.resize(grid, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite("/Users/piotrwojcik/PycharmProjects/Siamese-Image-Modeling/figs/grid_image.png", grid)

    # Display the grid image using OpenCV
    cv2.imshow("image", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def interleave_lists(*lists):
    max_length = max(len(lst) for lst in lists)
    interleaved = [val for pair in zip(*lists) for val in pair]

    for lst in lists:
        if len(lst) > max_length:
            interleaved += lst[max_length:]

    return interleaved


normalize = transforms.Compose([
    lambda x: x.float() / 255.0,
    lambda x: torch.einsum('chw->hwc', x),
])

denormalize = transforms.Compose([
    lambda x: torch.einsum('hwc->chw', x),
    lambda x: x * 255.0,
    lambda x: x.to(torch.uint8)
])


def draw_crop_boxes(images, crops):
    boxes = crops.clone()

    annotated_images = []

    for idx, image in enumerate(images):
        view1_box = boxes[idx, :4]
        view1_box[2:], view1_box[3] = view1_box[:2] + view1_box[2:], view1_box[1] + view1_box[3]
        view1_box = view1_box.unsqueeze(0)
        view1_box[:, [0, 1, 2, 3]] = view1_box[:, [1, 0, 3, 2]]

        view2_box = boxes[idx, 4:]
        view2_box[2:], view2_box[3] = view2_box[:2] + view2_box[2:], view2_box[1] + view2_box[3]
        view2_box = view2_box.unsqueeze(0)
        view2_box[:, [0, 1, 2, 3]] = view2_box[:, [1, 0, 3, 2]]

        views_boxes = torch.cat([view1_box, view2_box], dim=0)

        annotated_image = draw_bounding_boxes(denormalize(image), views_boxes, width=2, colors=["yellow", "green"])
        annotated_images.append(normalize(annotated_image))
    annotated_images = [img for img in annotated_images]
    return annotated_images


def draw_bboxes(images, boxes, classes):
    annotated_images = []

    boxes = boxes.float()

    nuclei_types = {0: "neo", 1: "inflam", 2: "conn", 3: "dead", 4: "epith"}

    for idx, image in enumerate(images):
        mask = torch.all(boxes[idx] != -1, dim=1)
        labels = [f"{i[0]}, {nuclei_types[i[1].item()]}" for i in enumerate(classes[idx, mask])]
        labels = np.array(labels).tolist()

        annotated_image = draw_bounding_boxes(denormalize(image), boxes[idx, mask, :], labels=labels,
                                              width=1, colors="red")
        annotated_images.append(normalize(annotated_image))

    annotated_images = [img for img in annotated_images]
    return annotated_images


def tensor_batch_to_list(tensor):
    tensor_list = [t for t in tensor]
    return tensor_list

@dataclass
class Config:
    data_path: str
    input_size: int
    num_boxes: int
    with_blockwise_mask: bool
    crop_min: float
    blockwise_num_masking_patches: int
    batch_size: int


args = Config(data_path='/Users/piotrwojcik/Downloads/fold_1_256_cls/positive/', input_size=224, with_blockwise_mask=True,
              blockwise_num_masking_patches=127, crop_min=0.2, num_boxes=150, batch_size=2)

if __name__ == '__main__':
    transform_train = DataAugmentationForSIMFinetune(args, is_training=True)
    print(f'Pre-train data transform:\n{transform_train}')

    dataset_finetune = ImgWithPickledBoxesAndClassesDataset(os.path.join(args.data_path),
                                                            ds_type='pannuke',
                                                            transform=transform_train)
    print(f'Build dataset: train images = {len(dataset_finetune)}')

    sampler_finetune = RandomSampler(dataset_finetune)
    dataloader_finetune = torch.utils.data.DataLoader(
        dataset_finetune, sampler=sampler_finetune,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )

    images = []

    for idx, data in enumerate(dataloader_finetune):
        samples = data
        x0 = samples['x0']
        x1 = samples['x1']
        boxes0 = samples['boxes0']
        boxes1 = samples['boxes1']
        classes = samples['classes']

        img0 = x0.permute(0, 2, 3, 1)
        img1 = x1.permute(0, 2, 3, 1)

        patch_size = 16

        #x_masked, mask, ids_restore = mask_mae(fake_embedding)
        #mask = mask.view(N, H // patch_size, W // patch_size)

        img0 = tensor_batch_to_list(img0)
        img1 = tensor_batch_to_list(img1)

        img0 = draw_bboxes(img0, boxes=boxes0, classes=classes)
        img1 = draw_bboxes(img1, boxes=boxes1, classes=classes)
        imgs = interleave_lists(img0, img1)
        images.extend(imgs)
        if idx == 1:
            break

    create_image_grid(images)

