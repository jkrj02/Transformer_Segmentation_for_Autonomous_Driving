import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import dataloader

train_path = 'option1_driving-segmentation/driving_train_data.h5'
test_path = 'option1_driving-segmentation/driving_test_data.h5'
Batch_Size = 1

if __name__ == '__main__':
    f = h5py.File(test_path)
    # for key in f.keys():
    #     print(f[key], key, f[key].name)
    # np_img = torchvision.utils.make_grid(f['rgb'][0])
    train_dataset = dataloader.LoadData(train_path, train=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=Batch_Size,
                                               shuffle=True)
    data_iter = iter(train_loader)
    images, label = next(data_iter)  # images are 32 x 1 x 28 x 28
    np_img = torchvision.utils.make_grid(images[0]).numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))  # H x W x color_channel
    plt.show()
    np_img_2 = label[0].numpy()
    plt.imshow(np.transpose(np_img_2, (1, 2, 0)))
    plt.show()