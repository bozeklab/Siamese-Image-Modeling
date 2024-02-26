import argparse
import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from matplotlib.colors import ListedColormap
from torch.utils.data import RandomSampler, SequentialSampler
from torchvision import transforms

from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.augmentation import GaussianBlur, SingleRandomResizedCrop, RandomHorizontalFlipBoxes, Solarize, \
    Resize, RandomHorizontalFlipForMaps, RandomHorizontalFlip
from util.datasets import ImagenetWithMask, ImagenetPlainWithMask
from main_pretrain import DataAugmentationBoxesForSIMTraining, DataAugmentationForSIM
from util.augmentation import SingleRandomResizedCrop
from util.datasets import ImagenetWithMask, ImagenetPlainWithMask
from util.unet import UNet, Adios_mask


@dataclass
class Config:
    data_path: str
    resnet_ckpt: str
    input_size: int
    num_boxes: int
    mask_fbase: int
    unet_norm: str
    with_blockwise_mask: bool
    crop_min: float
    blockwise_num_masking_patches: int
    batch_size: int


args = Config(data_path='/Users/piotrwojcik/TCGA_images',
              resnet_ckpt='/Users/piotrwojcik/PycharmProjects/Siamese-Image-Modeling/checkpoints/adios_simclr/simclr_adios_resnet18_he-ep_499.ckpt',
              mask_fbase=32, unet_norm='in',
              input_size=224, with_blockwise_mask=True,
              blockwise_num_masking_patches=127,
              crop_min=0.2, num_boxes=150, batch_size=1)


def bool_to_bw_image(bool_tensor):
    """
    Convert a boolean tensor to a black and white image.

    Parameters:
    - bool_tensor: The boolean tensor to convert.

    Returns:
    - bw_image: The black and white image tensor with shape (3, H, W).
    """
    h, w = bool_tensor.shape
    bw_image = torch.zeros(3, h, w, dtype=torch.uint8)

    # Set pixel values for True (1.0) and False (0.0)
    bw_image[0, bool_tensor] = 255  # Red channel
    bw_image[1, bool_tensor] = 255  # Green channel
    bw_image[2, bool_tensor] = 255  # Blue channel

    return bw_image


def masks_grid(input_tensor, patch_size=16):

    N, H, W = input_tensor.shape

    reshaped_tensor = input_tensor.view(N, H // patch_size, patch_size,
                                        W // patch_size, patch_size).contiguous()

    # Count the number of true values in each patch
    output_tensor = reshaped_tensor.sum(dim=(2, 4))

    return output_tensor


def threshold_grid(input_tensor, k, patch_size=16):

    # Reshape the input tensor to create non-overlapping patches of size 16x16
    reshaped_tensor = input_tensor.view(input_tensor.size(0) // patch_size, patch_size,
                                        input_tensor.size(1) // patch_size, patch_size).contiguous()

    # Count the number of true values in each patch
    count_tensor = reshaped_tensor.sum(dim=(1, 3))

    # Apply the threshold to create the boolean tensor
    output_tensor = count_tensor > k

    return output_tensor

def threshold_grid_batch(input_tensor, k, patch_size=16):

    # Reshape the input tensor to create non-overlapping patches of size 16x16
    reshaped_tensor = input_tensor.view(input_tensor.size(1) // patch_size, patch_size,
                                        input_tensor.size(2) // patch_size, patch_size).contiguous()

    # Count the number of true values in each patch
    count_tensor = reshaped_tensor.sum(dim=(2, 4))

    # Apply the threshold to create the boolean tensor
    output_tensor = count_tensor > k

    return output_tensor

def gray_out_square(image, x_start, y_start, size, alpha):
    # Get the dimensions of the image tensor
    _, height, width = image.shape

    # Calculate the end coordinates of the square region
    x_end = min(x_start + size, width)
    y_end = min(y_start + size, height)

    # Create a gray overlay image
    gray_overlay = alpha * image[:, x_start:x_end, y_start:y_end]

    # Replace the square region with the gray overlay
    image[:, x_start:x_end, y_start:y_end] = gray_overlay

    return image


def gray_out_mask(image, mask, patch_size, alpha):
    mh, mw = mask.shape

    for i in range(mh):
        for j in range(mw):
            if mask[i][j]:
                image = gray_out_square(image, i * patch_size, j * patch_size, patch_size, alpha)
    return image


def random_masking_setting3(mask_ratio, masks, total_num):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    """
    N, M, L = masks.shape  # batch, length, dim  [64,196,768]

    len_keep = int(L * (1 - mask_ratio))

    total_num = total_num.unsqueeze(2).repeat(1, 1, L)

    noise = torch.rand(N, 4, device=masks.device)  # noise in [0, 1]
    parts_id_shuffle = torch.argsort(noise, dim=1)
    parts_ids_restore = torch.argsort(parts_id_shuffle, dim=1)
    shuffle_total_num = torch.gather(total_num, dim=2, index=parts_id_shuffle.unsqueeze(2).repeat(1, 1, L))
    shuffle_total_num = shuffle_total_num.to(torch.long)
    marks = (L * mask_ratio - torch.cumsum(shuffle_total_num, -2) + shuffle_total_num).to(torch.long)  # mask_num-b+a
    marks_remains = torch.where(marks < 0, 0, marks).to(torch.long)  # max(mask_num-b+a, 0)
    mask_num = torch.where(marks_remains < shuffle_total_num, marks_remains,
                           shuffle_total_num)  # min(a, max(mask-b+a))
    mask_num = torch.gather(mask_num, dim=2, index=parts_ids_restore.unsqueeze(2).repeat(1, 1, L))
    #masks = torch.gather(masks, dim=2, index=parts_id_shuffle.unsqueeze(2).repeat(1, 1, L))

    cum_mask = torch.cumsum(masks, dim=2)

    judge_mask = cum_mask <= mask_num

    judge_mask = torch.sum(judge_mask * masks, dim=1)  # 0 is keep, 1 is remove
    judge_mask = torch.clamp(judge_mask, max=1)
    ids_shuffle = torch.argsort(judge_mask, dim=1)  # shuffle_id
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    # sorted_index = torch.gather(sorted_index, dim=1, index=ids_shuffle)
    # keep the first subset
    mask = torch.ones([N, L], device=masks.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask

    mask = torch.gather(mask, dim=1, index=ids_restore)
    return mask, ids_restore


class DataAugmentationForSIMSample(object):
    def __init__(self, args):
        self.args = args

        self.random_resized_crop = SingleRandomResizedCrop(args.input_size, scale=(args.crop_min, 1.0), interpolation=3)
        self.random_flip = RandomHorizontalFlip()

        self.color_transform1 = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
        ])

        self.color_transform2 = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            transforms.RandomApply([Solarize()], p=0.2),
        ])

        self.format_transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(
            #    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, image):
        spatial_image1, flip1 = self.random_flip(image)
        spatial_image2, flip2 = self.random_flip(image)
        spatial_image1, i1, j1, h1, w1, W = self.random_resized_crop(spatial_image1, boxes=None)
        spatial_image2, i2, j2, h2, w2, W = self.random_resized_crop(spatial_image2, boxes=None)

        relative_flip = (flip1 and not flip2) or (flip2 and not flip1)

        return {
            'x0': self.format_transform(image),
            'x1': self.format_transform(spatial_image1),
            'x2': self.format_transform(spatial_image2),
            'i1': i1,
            'i2': i2,
            'j1': j1,
            'j2': j2,
            'h1': h1,
            'h2': h2,
            'w1': w1,
            'w2': w2,
            'flip1': flip1,
            'flip2': flip2,
            'delta_i': (i2 - i1) / h1,
            'delta_j': (j2 - j1) / w1,
            'delta_h': h2 / h1,
            'delta_w': w2 / w1,
            'relative_flip': relative_flip,
            'flip_delta_j': (W - j1 - j2) / w1,
        }

    def __repr__(self):
        repr = "(DataAugmentation,\n"
        repr += "  transform = %s,\n" % str(self.random_resized_crop) + str(self.random_flip) + str(self.color_transform1) + str(self.format_transform)
        repr += ")"
        return repr


if __name__ == '__main__':
    transform_train = DataAugmentationForSIMSample(args)
    print(f'Pre-train data transform:\n{transform_train}')

    model = Adios_mask(
        num_blocks=int(np.log2(args.input_size) - 1),
        mask_fbase=args.mask_fbase,
        img_size=args.input_size,
        filter_start=args.mask_fbase,
        in_chnls=3,
        out_chnls=-1,
        norm=args.unet_norm,
        N=4)

    checkpoint = torch.load(args.resnet_ckpt, map_location='cpu')
    # load pre-trained model
    msg = model.load_state_dict(checkpoint['state_dict'], strict=False)
    print(msg)

    dataset_train = ImagenetPlainWithMask(os.path.join(args.data_path),
                                          input_size=args.input_size,
                                          transform=transform_train,
                                          with_blockwise_mask=args.with_blockwise_mask,
                                          blockwise_num_masking_patches=args.blockwise_num_masking_patches)

    print(f'Build dataset: train images = {len(dataset_train)}')

    sampler_train = RandomSampler(dataset_train)
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )

    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    images = []
    masked_images0 = []
    masked_images1 = []
    masked_images2 = []
    masked_images3 = []
    masked_images4 = []

    parts = []
    masks0 = []
    masks1 = []
    masks2 = []
    masks3 = []
    masks4 = []

    for idx, data in enumerate(dataloader_train):
        if idx == 5:
            break
        samples, mask = data
        x1 = samples['x1'].squeeze()

        model.eval()
        with torch.no_grad():
            x = x1.unsqueeze(dim=0)
            x = normalization(x)
            feats = model.mask_encoder(x)
            soft_masks = model.mask_head(feats)
            a = soft_masks.argmax(dim=1).cpu()
            hard_masks = torch.zeros(soft_masks.shape).scatter(1, a.unsqueeze(1), 1.0).squeeze()

            TH = 60

            # hard_masks_ = masks_grid(hard_masks)
            # background = torch.all(hard_masks_ <= TH, dim=0)
            # bckg = torch.where(background, torch.tensor(256.0), torch.tensor(0.0)).unsqueeze(0)
            # hard_masks_ = torch.concat([hard_masks_, bckg], dim=0)
            # output_tensor = hard_masks_.argmax(dim=0)
            # print(output_tensor)
            # #print(masks_grid(hard_masks).shape)
            # num_levels = 5
            # # Create a discrete colormap with 'viridis' as the base colormap
            # cmap = ListedColormap(plt.cm.viridis(np.linspace(0, 1, num_levels)))
            # plt.imshow(output_tensor, cmap=cmap, interpolation='nearest')
            # plt.colorbar(ticks=range(num_levels))
            # plt.show()

            #threshold_grid(hard_masks, TH)

            th0 = hard_masks[0].bool()
            th0 = threshold_grid(th0, TH)

            th1 = hard_masks[1].bool()
            th1 = threshold_grid(th1, TH)

            th2 = hard_masks[2].bool()
            th2 = threshold_grid(th2, TH)

            th3 = hard_masks[3].bool()
            th3 = threshold_grid(th3, TH)

            import random

            th = [th0, th1, th2, th3]
            random.shuffle(th)
            all_masks = torch.stack(th, dim=0)
            total_num = all_masks.sum(dim=(1, 2)).unsqueeze(dim=0)
            all_masks = all_masks.unsqueeze(dim=0).flatten(2)
            all_masks = all_masks.permute(0, 1, 2)

            smask, _ = random_masking_setting3(mask_ratio=0.5, masks=all_masks, total_num=total_num)

            smasks = smask.squeeze().view(th0.shape[0], th0.shape[1])

            #cum_mask = torch.bitwise_or.reduce(all_masks, dim=1)

            #count_per_row = torch.sum(all_masks, dim=(2, 3))
            #print(count_per_row)

            #B, N, H, W = all_masks.shape  # batch, length, dim  [64,196,768]
            #L = H * W
            #mask_ratio = 0.75
            #len_keep = int(L * (1 - mask_ratio))
            #noise = torch.rand(N, L)
            #ids_shuffle = torch.argsort(noise, dim=1)
            #print(ids_shuffle.shape)
            #all_masks = torch.gather(all_masks, dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, 4))

            #ids_shuffle = torch.argsort(noise, dim=1)

            x_g0 = gray_out_mask(x1.clone(), th0, 16, alpha=0.2)
            masked_images0.append(x_g0)
            x_g1 = gray_out_mask(x1.clone(), th1, 16, alpha=0.2)
            masked_images1.append(x_g1)
            x_g2 = gray_out_mask(x1.clone(), th2, 16, alpha=0.2)
            masked_images2.append(x_g2)
            x_g3 = gray_out_mask(x1.clone(), th3, 16, alpha=0.2)
            masked_images3.append(x_g3)
            x_g4 = gray_out_mask(x1.clone(), smasks, 16, alpha=0.2)
            masked_images4.append(x_g4)

            #x_g3 = gray_out_mask(x1.clone(), cum_mask, 16, alpha=0.2)
            #masked_images3.append(x_g3)

            m0 = bool_to_bw_image((1 - hard_masks[0]).bool())
            m1 = bool_to_bw_image((1 - hard_masks[1]).bool())
            m2 = bool_to_bw_image((1 - hard_masks[2]).bool())
            m3 = bool_to_bw_image((1 - hard_masks[3]).bool())

            masks0.append(m0.float())
            masks1.append(m1.float())
            masks2.append(m2.float())
            masks3.append(m3.float())

        images.append(x1)

    grid_masked_img0 = vutils.make_grid(masked_images0, nrow=len(masked_images0), padding=1, normalize=True)
    grid_masked_img1 = vutils.make_grid(masked_images1, nrow=len(masked_images1), padding=1, normalize=True)
    grid_masked_img2 = vutils.make_grid(masked_images2, nrow=len(masked_images2), padding=1, normalize=True)
    grid_masked_img3 = vutils.make_grid(masked_images3, nrow=len(masked_images3), padding=1, normalize=True)
    grid_masked_img4 = vutils.make_grid(masked_images4, nrow=len(masked_images4), padding=1, normalize=True)

    grid_tensor_images = vutils.make_grid(images, nrow=len(images), padding=1, normalize=True)
    grid_tensor_masks0 = vutils.make_grid(masks0, nrow=len(masks0), padding=1, normalize=True)
    grid_tensor_masks1 = vutils.make_grid(masks1, nrow=len(masks0), padding=1, normalize=True)
    grid_tensor_masks2 = vutils.make_grid(masks2, nrow=len(masks0), padding=1, normalize=True)
    grid_tensor_masks3 = vutils.make_grid(masks3, nrow=len(masks0), padding=1, normalize=True)

    # Convert the PyTorch tensors to NumPy arrays and transpose the dimensions
    grid_masked_img0 = grid_masked_img0.cpu().numpy().transpose((1, 2, 0))
    grid_masked_img1 = grid_masked_img1.cpu().numpy().transpose((1, 2, 0))
    grid_masked_img2 = grid_masked_img2.cpu().numpy().transpose((1, 2, 0))
    grid_masked_img3 = grid_masked_img3.cpu().numpy().transpose((1, 2, 0))
    grid_masked_img4 = grid_masked_img4.cpu().numpy().transpose((1, 2, 0))

    grid_images = grid_tensor_images.cpu().numpy().transpose((1, 2, 0))
    grid_masks0 = grid_tensor_masks0.cpu().numpy().transpose((1, 2, 0))
    grid_masks1 = grid_tensor_masks1.cpu().numpy().transpose((1, 2, 0))
    grid_masks2 = grid_tensor_masks2.cpu().numpy().transpose((1, 2, 0))
    grid_masks3 = grid_tensor_masks3.cpu().numpy().transpose((1, 2, 0))

    average_image0 = 0.6 * grid_images + 0.4 * grid_masks0
    average_image1 = 0.6 * grid_images + 0.4 * grid_masks1
    average_image2 = 0.6 * grid_images + 0.4 * grid_masks2
    average_image3 = 0.6 * grid_images + 0.4 * grid_masks3

    # Create a new figure with three subplots
    fig, axs = plt.subplots(10, 1, figsize=(8, 12))

    # Display the images in the first row
    axs[0].imshow(grid_images)
    axs[0].axis('off')

    ## Display the grayscale masks in the second row
    #axs[1].imshow(grid_masks0, cmap='gray')  # Use grayscale colormap
    #axs[1].axis('off')

    # Display the average of two images in the third row
    axs[1].imshow(average_image0)
    axs[1].axis('off')

    axs[2].imshow(average_image1)
    axs[2].axis('off')

    axs[3].imshow(average_image2)
    axs[3].axis('off')

    axs[4].imshow(average_image3)
    axs[4].axis('off')

    axs[5].imshow(grid_masked_img0)
    axs[5].axis('off')

    axs[6].imshow(grid_masked_img1)
    axs[6].axis('off')

    axs[7].imshow(grid_masked_img2)
    axs[7].axis('off')

    axs[8].imshow(grid_masked_img3)
    axs[8].axis('off')

    axs[9].imshow(grid_masked_img4)
    axs[9].axis('off')

    plt.show()
