import argparse
import os
import cv2
import numpy as np
import torch
from PIL.Image import Image

from torch.utils.data import RandomSampler, BatchSampler, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms as T
from torchvision.ops import masks_to_boxes

from main_pretrain import DataAugmentationForSIM
from util.datasets import ImagenetWithMask


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


def create_image_grid(images):
    # Determine the dimensions of each image in the grid
    rows, cols, _ = images[0].shape

    # Determine the number of images and columns in the grid
    num_images = len(images)
    num_cols = 3

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


def draw_bboxes(images, boxes):
    annotated_images = []

    # Create a mask for elements where the last positions are not all -1
    mask = (boxes[:, :, -1] != -1).unsqueeze(-1).expand_as(boxes)
    boxes = boxes.float()

    # Multiply the elements by a certain number (e.g., 2) only where the mask is True
    boxes[mask] *= 384/512

    for idx, image in enumerate(images):
        annotated_image = draw_bounding_boxes(denormalize(image), boxes[idx], width=2, colors="red")
        annotated_images.append(normalize(annotated_image))

    annotated_images = [img for img in annotated_images]
    return annotated_images


def tensor_batch_to_list(tensor):
    tensor_list = [t for t in tensor]
    return tensor_list


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=2, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=352, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    parser.add_argument('--use_abs_pos_emb', default=True, action='store_true')
    parser.add_argument('--disable_abs_pos_emb', dest='use_abs_pos_emb', action='store_false')
    parser.add_argument('--use_shared_rel_pos_bias', default=False, action='store_true')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/Users/piotrwojcik/images_he_seg1000/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    # parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # SiameseIM parameters
    # data
    parser.add_argument('--crop_min', default=0.2, type=float)
    parser.add_argument('--use_tcs_dataset', default=False, action='store_true')

    # model
    parser.add_argument('--decoder_embed_dim', default=512, type=int)
    parser.add_argument('--drop_path_rate', default=0.0, type=float)
    parser.add_argument('--init_values', default=None, type=float)
    parser.add_argument('--projector_depth', default=2, type=int)
    parser.add_argument('--predictor_depth', default=4, type=int)
    parser.add_argument('--use_proj_ln', default=False, action='store_true')
    parser.add_argument('--use_pred_ln', default=False, action='store_true')
    parser.add_argument('--train_patch_embed', default=False, action='store_true')
    parser.add_argument('--online_ln', default=False, action='store_true', help='also use frozen LN in online branch')

    parser.add_argument('--loss_type', default='mae')
    parser.add_argument('--neg_weight', default=0.02, type=float)

    parser.add_argument('--with_blockwise_mask', default=True, action='store_true')
    parser.add_argument('--blockwise_num_masking_patches', default=110 , type=int)

    # hyper-parameter
    parser.add_argument('--mm', default=0.996, type=float)
    parser.add_argument('--mmschedule', default='const')
    parser.add_argument('--lambda_F', default=50, type=float)  # may no need
    parser.add_argument('--T', default=0.2, type=float)  # check
    parser.add_argument('--clip_grad', default=None, type=float)
    parser.add_argument('--beta2', default=0.95, type=float)

    # misc
    parser.add_argument('--auto_resume', default=True)
    parser.add_argument('--save_freq', default=50, type=int)
    parser.add_argument('--save_latest_freq', default=1, type=int)
    parser.add_argument('--fp32', default=False, action='store_true')
    parser.add_argument('--amp_growth_interval', default=2000, type=int)

    return parser


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    transform_train = DataAugmentationForSIM(args)
    print(f'Pre-train data transform:\n{transform_train}')

    dataset_train = ImagenetWithMask(os.path.join(args.data_path),
                                     input_size=args.input_size,
                                     transform=transform_train,
                                     with_blockwise_mask=args.with_blockwise_mask,
                                     blockwise_num_masking_patches=args.blockwise_num_masking_patches)
    print(f'Build dataset: train images = {len(dataset_train)}')

    sampler_train = RandomSampler(dataset_train)
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    images = []

    for idx, data in enumerate(dataloader_train):
        samples, labels, mask = data
        x0 = samples['x0']
        x1 = samples['x1']
        x2 = samples['x2']
        delta_i = samples['delta_i']
        delta_j = samples['delta_j']
        delta_h = samples['delta_h']
        delta_w = samples['delta_w']
        relative_flip = samples['relative_flip']
        flip_delta_j = samples['flip_delta_j']

        img0 = x0.permute(0, 2, 3, 1)
        img1 = x1.permute(0, 2, 3, 1)
        img2 = x2.permute(0, 2, 3, 1)

        patch_size = 16
        N, H, W, C = img0.shape
        fake_embedding = torch.rand((N, (H // patch_size) * (W // patch_size), C))

        #x_masked, mask, ids_restore = mask_mae(fake_embedding)
        #mask = mask.view(N, H // patch_size, W // patch_size)

        img0 = tensor_batch_to_list(img0)
        img1 = tensor_batch_to_list(img1)
        img2 = tensor_batch_to_list(img2)

        mask = tensor_batch_to_list(mask)

        #img0 = draw_crop_boxes(img0, pos)
        #img0 = draw_bboxes(img0, boxes)
        img1 = [gray_out_mask(img, mask, patch_size, alpha=0.5) for img, mask in zip(img1, mask)]
        imgs = interleave_lists(img0, img1, img2)
        images.extend(imgs)
        if idx == 1:
            break

    create_image_grid(images)

