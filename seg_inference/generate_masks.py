import os
from dataclasses import dataclass

import torch
from imgaug import SegmentationMapsOnImage
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler

import imageio
import imgaug as ia
from engine_finetune_unetr import unpack_predictions, unpack_masks
from main_finetune_unetr import prepare_model
from main_pretrain import DataAugmentationForImagesWithMaps
from util.img_with_mask_dataset import ImagesWithSegmentationMaps, PanNukeDataset
from util.metrics import remap_label
from util.pos_embed import interpolate_pos_embed


@dataclass
class Config:
    data_path: str
    encoder_path: str
    model_path: str
    input_size: int
    batch_size: int
    embed_dim: int
    drop_path: float


args = Config(data_path='/Users/piotrwojcik/data/pannuke/fold2',
              encoder_path='/Users/piotrwojcik/PycharmProjects/Siamese-Image-Modeling/seg_inference/checkpoints/checkpoint-49.pth',
              model_path='/Users/piotrwojcik/PycharmProjects/Siamese-Image-Modeling/seg_inference/checkpoints/unetr_pt.pth.tar',
              input_size=256,
              embed_dim=768,
              drop_path=0.1,
              batch_size=2)

if __name__ == '__main__':
    transform_mask = DataAugmentationForImagesWithMaps(False, args)

    print(f'Segmentation data transform:\n{transform_mask}')

    mask_eval = PanNukeDataset(os.path.join(args.data_path), transform=transform_mask)

    random_sampler = SequentialSampler(mask_eval)
    dataloader = DataLoader(mask_eval, batch_size=args.batch_size, num_workers=2, sampler=random_sampler)

    print(f'Build dataset: images with mask = {len(mask_eval)}')

    num_nuclei_classes = len(PanNukeDataset.nuclei_types)
    num_tissue_classes = len(PanNukeDataset.tissue_types)
    #print(num_nuclei_classes)
    #print(num_tissue_classes)
    encoder_model = prepare_model(args.encoder_path,
                                  init_values=None,
                                  drop_path_rate=args.drop_path,
                                  num_nuclei_classes=num_nuclei_classes,
                                  num_tissue_classes=num_tissue_classes,
                                  embed_dim=args.embed_dim,
                                  extract_layers=[3, 6, 9, 12])

    checkpoint = torch.load(args.model_path, map_location='cpu')
    pretrained_dict = {k.replace('module.encoder.', ''): v for k, v in checkpoint['model'].items()}

    branches = ['nuclei_binary_map_decoder', 'hv_map_decoder', 'nuclei_type_maps_decoder']
    _pretrained_dict = []
    for k, v in pretrained_dict.items():
        if k.startswith('common_decoder'):
            for b in branches:
                nb = k.replace('common_decoder', b)
                _pretrained_dict.append((nb, v.clone()))
                #print(nb)
        else:
            _pretrained_dict.append((k, v))
    pretrained_dict = dict(_pretrained_dict)
    msg = encoder_model.load_state_dict(pretrained_dict, strict=False)
    print(msg)

    checkpoint_encoder = torch.load(args.encoder_path, map_location='cpu')
    checkpoint_encoder = checkpoint_encoder['model']
    for k in checkpoint_encoder.keys():
        v1 = checkpoint_encoder[k]
        v2 = encoder_model.state_dict()[f"encoder.{k}"]
        if not torch.equal(v1, v2):
            print('!!!')
            print(k)
    print('all equal 2')

    encoder_model.eval()

    for idx, sample in enumerate(dataloader):
        cells = []

        if idx == 20:
            break

        x0 = sample['x0']
        x = sample['x']

        predictions_ = encoder_model(x0)
        predictions = unpack_predictions(predictions_, num_nuclei_classes, x.device)
        gt = unpack_masks(masks=sample, device=x.device, tissues_map=PanNukeDataset.tissue_types,
                          num_nuclei_classes=num_nuclei_classes)

        x0 = x0[0]
        x = x[0]

        x0 = torch.einsum('chw -> hwc', x0)
        x0 = (x0 * 255.0).to(torch.uint8).detach().numpy()

        x = torch.einsum('chw -> hwc', x)
        x = (x * 255.0).to(torch.uint8).detach().numpy()

        cells.append(x0)  # column 1
        cells.append(x)  # column 2


        imap = predictions['instance_map'][0].detach().numpy()
        imap = remap_label(imap)
        tmap = predictions['nuclei_type_map'][0].detach().numpy()

        imap_gt = gt['instance_map'][0].detach().numpy()
        imap_gt = remap_label(imap_gt)
        tmap_gt = gt['nuclei_type_map'][0].detach().numpy()

        imap = SegmentationMapsOnImage(imap, shape=x.shape)
        tmap = SegmentationMapsOnImage(tmap, shape=x.shape)
        imap_gt = SegmentationMapsOnImage(imap_gt, shape=x.shape)
        tmap_gt = SegmentationMapsOnImage(tmap_gt, shape=x.shape)

        cells.append(imap.draw_on_image(x)[0])  # column 3
        cells.append(tmap.draw_on_image(x)[0])  # column 4
        cells.append(imap_gt.draw_on_image(x)[0])  # column 3
        cells.append(tmap_gt.draw_on_image(x)[0])  # column 4

        cells.append(tmap.draw(size=x.shape[:2])[0])  # column 5
        cells.append(tmap_gt.draw(size=x.shape[:2])[0])  # column 6

        # Convert cells to a grid image and save.
        grid_image = ia.draw_grid(cells, cols=8)
        imageio.imwrite(f"eve_test_{idx}.jpg", grid_image)
