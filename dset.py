import h5py
from PIL import Image
import torch
from torchvision import transforms
import settings

all_data = h5py.File(settings.DIR, 'r')


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])




class MY_DATASET(torch.utils.data.Dataset):
    #
    def __init__(self, transform=None, target_transform=None, train=True, database=False):
        self.transform = transform
        self.target_transform = target_transform

        if train:
            self.train_labels = all_data['L_tr'][:].T
            self.txt = all_data['T_tr'][:].T
            self.images = all_data['I_tr'][:].transpose(3, 0, 1, 2)

        elif database:
            self.train_labels = all_data['L_db'][:].T
            self.txt = all_data['T_db'][:].T
            self.images = all_data['I_db'][:].transpose(3, 0, 1, 2)

        else:
            self.train_labels = all_data['L_te'][:].T
            self.txt = all_data['T_te'][:].T
            self.images = all_data['I_te'][:].transpose(3, 0, 1, 2)




    # if use wiki dataset
    # def __init__(self, transform=None, target_transform=None, train=True, database=False):
    #     self.transform = transform
    #     self.target_transform = target_transform
    #     images = all_data['IAll'][:]
    #     labels = all_data['LAll'][:]
    #     tags = all_data['TAll'][:]
    #     images = images.transpose(3, 1, 0, 2)
    #     labels = labels.transpose(1, 0)
    #     tags = tags.transpose(1, 0)
    #     query_size = 693
    #     training_size = 2173
    #     database_size = 2866
    #     if train:
    #         self.images = images[query_size: training_size + query_size]
    #         self.txt = tags[query_size: training_size + query_size]
    #         self.train_labels = labels[query_size: training_size + query_size]
    #     elif database:
    #         self.images = images
    #         self.txt = tags
    #         self.train_labels = labels
    #     else:
    #         self.images = images[0: query_size]
    #         self.txt = tags[0: query_size]
    #         self.train_labels = labels[0: query_size]


    def __getitem__(self, index):


        img, target = self.images[index, :,:,:], self.train_labels[index]


        img = Image.fromarray(img) # HWC

        txt = self.txt[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, txt, target, index

    def __len__(self):

        return len(self.train_labels)

