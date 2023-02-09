from torch.utils.data import dataset


class LoadData(dataset.Dataset):
    def __init__(self, path, train=True):
        super(LoadData, self).__init__()
        if train:
            

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
