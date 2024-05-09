import os
import glob
import torch
from PIL import Image
import pandas as pd

class SkinDataset(torch.utils.data.Dataset):
    def __init__(self, root, csv_file, transform=None, train=True, val=False,
                tumor=False):
        csv_file = os.path.join(root, csv_file)
        self.data = pd.read_csv(csv_file)

        if train:
            self.data = self.data[self.data['set'] == 'Train']
        else:
            if val:
                self.data = self.data[self.data['set'] == "Validation"]
            else:
                self.data = self.data[self.data['set'] == 'Test']

        if tumor:
            self.data = self.data[self.data['malignicy'] == 'tumor']
        self.tumor = tumor

        self.image_paths = self.data['file'].values
        self.labels = self.data['class'].values
        
        root = os.path.normpath(root)
        if 'data' in os.path.split(root)[-1]:
            self.root = os.path.join(os.path.split(root)[0])
        
        self.transform = transform
        self.train = train

        self.cat_to_num_map = {'nontumor_skin_necrosis_necrosis': 0,
                                'nontumor_skin_muscle_skeletal': 1,
                                'nontumor_skin_sweatglands_sweatglands': 2,
                                'nontumor_skin_vessel_vessel': 3,
                                'nontumor_skin_elastosis_elastosis': 4,
                                'nontumor_skin_chondraltissue_chondraltissue': 5,
                                'nontumor_skin_hairfollicle_hairfollicle': 6,
                                'nontumor_skin_epidermis_epidermis': 7,
                                'nontumor_skin_nerves_nerves': 8,
                                'nontumor_skin_subcutis_subcutis': 9,
                                'nontumor_skin_dermis_dermis': 10,
                                'nontumor_skin_sebaceousglands_sebaceousglands': 11,
                                'tumor_skin_epithelial_sqcc': 12,
                                'tumor_skin_melanoma_melanoma': 13,
                                'tumor_skin_epithelial_bcc': 14,
                                'tumor_skin_naevus_naevus': 15
                                }

        self.tumor_map = {'tumor_skin_epithelial_sqcc': 0,
                            'tumor_skin_melanoma_melanoma': 1,
                            'tumor_skin_epithelial_bcc': 2,
                            'tumor_skin_naevus_naevus': 3
                            }

        self.classes = list(self.cat_to_num_map)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        image_path = os.path.join(self.root, self.image_paths[index])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if not self.tumor:
            label = self.cat_to_num_map[self.labels[index]]
        else:
            label = self.tumor_map[self.labels[index]]

        return image, label