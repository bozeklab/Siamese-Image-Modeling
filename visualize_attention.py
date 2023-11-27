# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
import matplotlib.pyplot as plt
from io import BytesIO

import skimage.io
from matplotlib.colors import ListedColormap
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

import util
import models_dino as dino
import models_sim
from attnmask import AttMask, get_pred_ratio


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--model', default='sim_vit_base_patch16_img224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--pretrained_weights', default='checkpoints/checkpoint-49.pth', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument("--checkpoint_key", default="model", type=str,
        help='Key to use in the checkpoint (example: "model")')
    parser.add_argument("--image_size", default=(224, 224), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='debug/', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=0.75, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    # model
    parser.add_argument('--decoder_embed_dim', default=768, type=int)
    parser.add_argument('--drop_path_rate', default=0.0, type=float)
    parser.add_argument('--pred_shape', default='attmask_high', type=str, help="""Shape of partial prediction. 
                        Select between attmask_high, attmask_hint, attmask_low, rand or block""")
    parser.add_argument('--init_values', default=None, type=float)
    parser.add_argument('--projector_depth', default=2, type=int)
    parser.add_argument('--predictor_depth', default=4, type=int)

    # Attention parameters
    parser.add_argument('--masking_prob', type=float, default=0.7, help=""""Perform token masking 
                        based on attention with specific probability, works only for --pred_shape attmask_high, attmask_hint, attmask_low""")
    parser.add_argument('--show_max', type=float, default=0.1,
                        help="The top salient tokens from which a random sample will be revealed")

    parser.add_argument('--use_proj_ln', default=False, action='store_true')
    parser.add_argument('--loss_type', default='mae')
    parser.add_argument('--use_pred_ln', default=False, action='store_true')
    parser.add_argument('--train_patch_embed', default=False, action='store_true')
    parser.add_argument('--online_ln', default=False, action='store_true', help='also use frozen LN in online branch')
    parser.add_argument('--use_abs_pos_emb', default=True, action='store_true')
    parser.add_argument('--disable_abs_pos_emb', dest='use_abs_pos_emb', action='store_false')
    parser.add_argument('--use_shared_rel_pos_bias', default=False, action='store_true')

    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    model = model = models_sim.__dict__[args.model](norm_pix_loss=False, args=args)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]

        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    else:
        print("There is no reference weights available for this model => We use random weights.")


    #if os.path.isfile(args.pretrained_weights):
    #    state_dict = torch.load(args.pretrained_weights, map_location="cpu")
    #    if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
    #        print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
    #        state_dict = state_dict[args.checkpoint_key]
    #        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

    #    msg = model.load_state_dict(state_dict, strict=False)
    #    print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    #else:
    #    print("There is no reference weights available for this model => We use random weights.")

    # open image
    if args.image_path is None:
        # user has not specified any image - we use our own image
        print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
        print("Since no image path have been provided, we take the first image in our paper.")
        response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
    elif os.path.isfile(args.image_path):
        with open(args.image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
    else:
        print(f"Provided image path {args.image_path} is non valid.")
        sys.exit(1)
    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size),
        pth_transforms.ToTensor(),
    ])
    img = transform(img)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // args.patch_size
    h_featmap = img.shape[-1] // args.patch_size

    attentions = model.get_last_selfattention(img.to(device))

    cls_attention = attentions[:, :, 0, 1:].mean(1).detach().clone()

    # Get AttMask. cls_attention should be in shape (batch_size, number_of_tokens)
    masks = AttMask(cls_attention,
                    args.masking_prob,
                    args.pred_shape,
                    get_pred_ratio(),
                    # For each sample in the batch we perform the same masking ratio
                    args.show_max * get_pred_ratio(),
                    args.show_max)

    masks = masks.reshape(-1, args.image_size[0] // args.patch_size, args.image_size[1] // args.patch_size).squeeze()
    print(masks)

    # Convert the boolean tensor to a float tensor
    float_tensor = masks.float()

    # Define a custom colormap with two colors (e.g., white and red)
    colors = ['white', 'red']
    cmap = ListedColormap(colors)

    # Display the heatmap with the custom colormap
    plt.imshow(float_tensor, cmap=cmap, interpolation='nearest')
    plt.title('Boolean Tensor Heatmap')
    plt.colorbar(ticks=[0, 1], format="%g", orientation='vertical')
    plt.show()

    with open(args.image_path, 'rb') as f:
        imgo = Image.open(f)
        imgo = imgo.convert('RGB')
        imgo = transform(imgo)
    img1 = gray_out_mask(imgo, masks, 16, alpha=0.5)
    from torchvision import transforms
    to_pil_transform = transforms.ToPILImage()
    to_pil_transform(img1).show()

    nh = attentions.shape[1] # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    if args.threshold is not None:
        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - args.threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()


    # save attentions heatmaps
    os.makedirs(args.output_dir, exist_ok=True)
    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(args.output_dir, "img.png"))
    for j in range(nh):
        fname = os.path.join(args.output_dir, "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        print(f"{fname} saved.")

    if args.threshold is not None:
        image = skimage.io.imread(os.path.join(args.output_dir, "img.png"))
        for j in range(nh):
            display_instances(image, th_attn[j], fname=os.path.join(args.output_dir, "mask_th" + str(args.threshold) + "_head" + str(j) +".png"), blur=False)