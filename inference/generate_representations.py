import os
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm

from main_pretrain import DataPreprocessingForSIM
from models_vit import vit_base_patch16
from PIL import Image
import torchvision.transforms.functional as F
from util.img_with_pickle_dataset import ImgWithPickledBoxesAndClassesDataset

import pickle

@dataclass
class Config:
    data_path: str
    input_size: int
    num_boxes: int
    batch_size: int
    init_values: float
    drop_path: float


args = Config(data_path='/Users/piotrwojcik/data/he/positive',
              input_size=352,
              num_boxes=350,
              batch_size=1,
              init_values=None,
              drop_path=0.0)


def prepare_model(chkpt_dir, **kwargs):
    # build model
    model = vit_base_patch16(**kwargs)

    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


if __name__ == '__main__':
    transform_inference = DataPreprocessingForSIM(args)
    print(f'Data pre-processing:\n{transform_inference}')

    dataset_inference = ImgWithPickledBoxesAndClassesDataset(os.path.join(args.data_path), transform=transform_inference)

    print(f'Build dataset: inference images = {len(dataset_inference)}')

    model_sim = prepare_model('checkpoints/checkpoint-latest.pth',
                              init_values=args.init_values,
                              global_pool=True,
                              drop_path_rate=args.drop_path,
                              box_patch_size=8)

    model_sim.eval()

    reps = []
    cls = []
    crops = []

    for idx, sample in tqdm(enumerate(dataset_inference), total=len(dataset_inference)):
        image = sample['x']

        boxes = sample['boxes']
        classes = sample['classes']
        x = image.unsqueeze(dim=0)
        boxes = boxes.unsqueeze(dim=0)
        classes = classes.unsqueeze(dim=0)
        box_features = model_sim.forward_boxes(x=x, boxes=boxes)
        mask = classes != -1
        classes = classes[mask]
        reps.append(box_features)
        cls.append(classes)

        for j in range(box_features.shape[0]):
            box = boxes[0, j].numpy().tolist()
            crop = image[:, int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            crop = F.resize(crop, size=(32, 32))
            crop = (crop * 255.0).to(torch.uint8)
            crops.append(crop.permute(1, 2, 0).numpy())

    X = torch.cat(reps, dim=0).numpy()
    y = torch.cat(cls, dim=0).numpy().astype(str)
    print('Finished inference...')

    output_file = "representations/data.pkl"

    data = {
     'X': X,
     'y': y,
     'crops': crops
    }

    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

    print('Data saved to: ', output_file)

