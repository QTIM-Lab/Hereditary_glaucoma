"""Dataset class for the Chaksu dataset."""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class ChaksuDataset(Dataset):
    def __init__(self, img_folder: str, label_file:str, split=None, transform=None):
        self.img_folder = img_folder
        self.transform = transform

        # read label file
        self.data = pd.read_csv(label_file)
        # Test set for Forus and Remidio, majority decision is called glaucoma decision
        self.data = self.data.rename(columns={'Glaucoma Decision': 'Majority Decision'})
        self.data['label'] = self.data['Majority Decision'].apply(lambda x: int(x == 'GLAUCOMA SUSPECT'))
        self.data['image_name'] = self.data['Images'].apply(lambda x: os.path.splitext(os.path.join(self.img_folder, x.split('-')[0]))[0])

        # there's more images in the labels than in the folder (?) - create a DataFrame with images in the folder and merge it with labels info
        self.img_paths = pd.DataFrame({'image_name': os.listdir(self.img_folder)})
        self.img_paths = self.img_paths[self.img_paths['image_name'].apply(lambda x: os.path.splitext(x)[1].lower() in ['.jpg', '.jpeg', '.png'])]
        self.img_paths['path'] = self.img_paths['image_name'].apply(lambda x: os.path.join(self.img_folder, x))
        self.img_paths['filename'] = self.img_paths['path'].apply(lambda x: os.path.splitext(x)[0])

        self.data = pd.merge(self.img_paths[['path', 'filename']], self.data, left_on='filename', right_on='image_name', how='left')
        
        # sanity check
        assert self.data['path'].apply(lambda x: os.path.exists(x)).all(), "Some images do not exist in the image folder [{}]".format(self.img_folder)

        # not used for now
        if split in ['train', 'val']:
            val = self.data.sample(frac=0.2, random_state=42)
            train = self.data[~self.data.index.isin(val.index)]
            self.data = train if split == 'train' else val

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]

        # load image
        img = Image.open(sample['path']).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return {'img': img, 'label': torch.tensor([sample['label']])}
