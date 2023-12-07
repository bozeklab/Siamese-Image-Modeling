import os

import torch
import torchvision.datasets as datasets
from PIL import Image
import pickle


class ImgDataset(datasets.VisionDataset):
    def __init__(self, root, transform):
        super(ImgDataset, self).__init__(root, transform=transform)

        self.root = root
        self.transform = transform

        self.file_list = [filename for filename in os.listdir(self.root) if filename.endswith('.png')]

    def __getitem__(self, index):
        filename = self.file_list[index]
        image_path = os.path.join(self.root, filename)

        image = Image.open(image_path).convert('RGB')
        return self.transform(image)

    def __len__(self):
        return len(self.file_list)


class ImgWithPickledBoxesDataset(datasets.VisionDataset):
    def __init__(self, root, transform):
        super(ImgWithPickledBoxesDataset, self).__init__(root, transform=transform)

        self.root = root
        self.transform = transform

        self.file_list = [filename for filename in os.listdir(self.root) if filename.endswith('.png')]

    def __getitem__(self, index):
        filename = self.file_list[index]
        image_path = os.path.join(self.root, filename)
        pickle_path = os.path.join(self.root, os.path.splitext(filename)[0] + '.pkl')

        image = Image.open(image_path).convert('RGB')
        with open(pickle_path, 'rb') as pickle_file:
            pickle_data = pickle.load(pickle_file)

        return self.transform(image, pickle_data)

    def __len__(self):
        return len(self.file_list)


dlbcl_cells = {
    'plasma_cell': 0,
    'eosinophil': 1,
    'macrophage': 2,
    'vessel': 3,
    'apoptotic_bodies': 4,
    'epithelial_cell': 5,
    'normal_small_lymphocyte': 6,
    'large_leucocyte': 7,
    'stroma': 8,
    'immune_cells': 9,
    'unknown': 10,
    'erythrocyte': 11,
    'mitose': 12,
    'positive': 13,
    'tumor': 14
}


class ImgWithPickledBoxesAndClassesDataset(datasets.VisionDataset):
    def __init__(self, root, transform, ds_type='dlbcl'):
        super(ImgWithPickledBoxesAndClassesDataset, self).__init__(root, transform=transform)

        self.root = root
        self.transform = transform
        self.cls_loader = ds_type

        self.file_list = [filename for filename in os.listdir(self.root) if filename.endswith('.png')]

    def __getitem__(self, index):
        filename = self.file_list[index]
        image_path = os.path.join(self.root, filename)
        box_path = os.path.join(self.root, os.path.splitext(filename)[0] + '.pkl')
        class_path = os.path.join(self.root, os.path.splitext(filename)[0] + '_cls.pkl')

        image = Image.open(image_path).convert('RGB')
        with open(box_path, 'rb') as box_file:
            box_data = pickle.load(box_file)
        with open(class_path, 'rb') as class_file:
            cls = pickle.load(class_file)
            if self.cls_loader in ['dlbcl',]:
                class_data = torch.tensor([dlbcl_cells[s] for s in cls[0]])
            elif self.cls_loader in ['pannuke',]:
                class_data = cls

        return self.transform(image, box_data, class_data)

    def __len__(self):
        return len(self.file_list)


