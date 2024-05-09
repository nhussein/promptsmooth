import os
import glob
import torch
from PIL import Image
import pandas as pd


class SicapDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_dir, transform=None, train=True):

        image_dir = os.path.join(root, image_dir)

        if train:
            csv_file = os.path.join(root, "partition/Test", "Train.xlsx")
            self.data = pd.read_excel(csv_file)
        else:
            csv_file = os.path.join(root, "partition/Test", "Test.xlsx")
            self.data = pd.read_excel(csv_file)

        # drop all columns except image_name and the label columns
        label_columns = ['NC', 'G3', 'G4', 'G5']  # , 'G4C']
        self.data = self.data[['image_name'] + label_columns]

        # get the index of the maximum label value for each row
        self.data['labels'] = self.data[label_columns].idxmax(axis=1)

        # replace the label column values with categorical values
        self.cat_to_num_map = label_map = {'NC': 0, 'G3': 1, 'G4': 2, 'G5': 3}  # , 'G4C': 4}
        self.data['labels'] = self.data['labels'].map(label_map)

        self.image_paths = self.data['image_name'].values
        self.labels = self.data['labels'].values
        self.image_dir = image_dir
        self.transform = transform
        self.train = train
        self.classes = ["non-cancerous well-differentiated glands",
                        "gleason grade 3 with atrophic well differentiated and dense glandular regions",
                        "gleason grade 4 with cribriform, ill-formed, large-fused and papillary glandular patterns",
                        "gleason grade 5 with nests of cells without lumen formation, isolated cells and pseudo-roseting patterns",
                        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_paths[index])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[index]

        return image, label