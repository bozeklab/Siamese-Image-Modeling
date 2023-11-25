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
from io import BytesIO

import skimage.io
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
import models_sim


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


def evaluate_best_head(attentions: torch.Tensor, bs: int, w_featmap: int, h_featmap: int, patch_size: int,
                       maps: torch.Tensor, threshold=0.6) -> torch.Tensor:
    jacs = 0
    nh = attentions.shape[1] # number of heads

    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=-1, keepdim=True)
    cumval = torch.cumsum(val, dim=-1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[:, head] = torch.gather(th_attn[:, head], dim=1, index=idx2[:, head])
    th_attn = th_attn.reshape(bs, nh, w_featmap, h_featmap).float()
    # interpolate
    th_attn = nn.functional.interpolate(th_attn, scale_factor=patch_size, mode="nearest").cpu().numpy()

    # Calculate IoU for each image
    for k, map in enumerate(maps):
        jac = 0
        objects = np.unique(map)
        objects = np.delete(objects, [-1])
        for o in objects:
            masko = map == o
            intersection = masko * th_attn[k]
            intersection = torch.sum(torch.sum(intersection, dim=-1), dim=-1)
            union = (masko + th_attn[k]) > 0
            union = torch.sum(torch.sum(union, dim=-1), dim=-1)
            jaco = intersection / union
            jac += max(jaco)
        if len(objects) != 0:
            jac /= len(objects)
        jacs += jac
    return torch.tensor(jacs)


CMAP = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (255, 0, 255),
    5: (0, 0, 0),
}


class RGBImageToTensor(object):
    def __init__(self, color_map=CMAP):
        self.color_map = color_map

    def __call__(self, img):
        # Convert PIL Image to NumPy array
        rgb_array = np.array(img)

        # Create a 2D tensor by mapping RGB values to integers using the color map
        height, width, _ = rgb_array.shape
        tensor_2d = np.zeros((height, width), dtype=np.uint8)

        for i, color in self.color_map.items():
            mask = np.all(rgb_array == np.array(color), axis=-1)
            tensor_2d[mask] = i

        float_tensor = torch.from_numpy(tensor_2d).float() / float(len(self.color_map) - 1)

        return float_tensor.unsqueeze(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--model', default='sim_vit_base_patch16_img224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--pretrained_weights', default='checkpoints/checkpoint-49.pth', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--voc_dir", default='/Users/piotrwojcik/Downloads/PanNukeVOC/', type=str, help="Path of the image directory to load.")
    parser.add_argument("--checkpoint_key", default="model", type=str,
        help='Key to use in the checkpoint (example: "model")')
    parser.add_argument("--image_size", default=(224, 224), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='debug/', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=0.6, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    # model
    parser.add_argument('--decoder_embed_dim', default=768, type=int)
    parser.add_argument('--drop_path_rate', default=0.0, type=float)
    parser.add_argument('--init_values', default=None, type=float)
    parser.add_argument('--projector_depth', default=2, type=int)
    parser.add_argument('--predictor_depth', default=4, type=int)
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
        #state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        #if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
        #    print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
        #    state_dict = state_dict[args.checkpoint_key]
        #msg = model.load_state_dict(state_dict, strict=False)

        checkpoint = torch.load(args.pretrained_weights, map_location='cpu')['teacher']
        pretrained_dict = {k.replace('backbone.', ''): v for k, v in checkpoint.items()}
        msg = model.load_state_dict(pretrained_dict, strict=False)
        print("Loaded checkpoint: ", msg)

        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    else:
        print("There is no reference weights available for this model => We use random weights.")

    # open image
    image_files = [f for f in os.listdir(os.path.join(args.voc_dir, 'Images')) if f.endswith('.png')]
    masks_files = [f for f in os.listdir(os.path.join(args.voc_dir, 'Masks')) if f.endswith('.png')]
    # Filter the list to include only PNG files
    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size),
        pth_transforms.ToTensor(),
    ])

    m_transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size),
        RGBImageToTensor(),
    ])
    jacs_all_heads = 0
    for idx in range(len(image_files)):

        image_file = image_files[idx]
        mask_file = masks_files[idx]

        image_path = os.path.join(args.voc_dir, 'Images', image_file)
        mask_path = os.path.join(args.voc_dir, 'Masks', mask_file)
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img = transform(img)

        with open(mask_path, 'rb') as f:
            mask = Image.open(f)
            mask = mask.convert('RGB')
            mask = m_transform(mask)

        img = img.unsqueeze(0)
        mask = mask.unsqueeze(0)

        patch_size = 16
        w_featmap = img.shape[-2] // patch_size
        h_featmap = img.shape[-1] // patch_size

        with torch.no_grad():
            attentions = model.get_last_selfattention(img)
        bs = attentions.shape[0]
        attentions = attentions[..., 0, 1:]

        d = evaluate_best_head(attentions, bs, w_featmap, h_featmap, patch_size, mask)
        if idx % 10 == 0:
            print('Evaluation: ', idx, ' ', d.item())
        jacs_all_heads += d
    jacs_all_heads /= len(image_files)
    print(f"All heads Jaccard on VOC12: {jacs_all_heads.item()}")


