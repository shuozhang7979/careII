import os
import h5py
from data.base_dataset import BaseDataset, get_params, get_transform
from data.k_folder import DataList
import torchvision.transforms as transforms
import numpy as np


class CombingMaskDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """
    def __init__(self, opt, data_list):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_path = opt.dataset_root  # get the image directory
        self.h5_list = data_list

    def __getitem__(self, index):
        """Return a slice and its mask.
        Parameters:
            index - - a random integer for data indexing
        Returns a list that contains A, B
            labeled_slice (tensor) - - an image in the input domain (720*720)
            mask  (tensor) - - its corresponding image in the target domain (720*720*2)
        """
        # read a image given a random integer index
        h5_path = self.h5_list[index]
        f = h5py.File(h5_path, 'r')
        labeled_slice = np.array(f['Dicom']).astype(np.float32)
        mask = np.array(f['combing_mask']).astype(np.float32)

        # apply the same transform to both A and B
        # transform_params = get_params(self.opt, A.size)
        #
        # A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        # B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        #
        # A = A_transform(A)
        # B = B_transform(B)
        transform_list = [transforms.ToTensor(), transforms.CenterCrop((512, 512))]
        trans = transforms.Compose(transform_list)
        labeled_slice = trans(labeled_slice)
        mask = trans(mask)
        return labeled_slice.cuda(), mask.cuda(), h5_path

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return len(self.h5_list)
