import os

import cv2
import h5py
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        f = h5py.File(path)
        self.data = f['rgb']  # type dataset
        self.label = f['seg']  # inside type nparray
        self.train = train

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = cv2.imread(os.path.join(self.img_dir,
                      img_id + self.img_ext))
        mask = cv2.imread(os.path.join(self.mask_dir,
                    img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None]
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32')
        mask = mask.transpose(2, 0, 1)
        
        return img, mask, {'img_id': img_id}
