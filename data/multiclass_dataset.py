import os
import h5py
from data.base_dataset import BaseDataset, get_params, get_transform
from data.k_folder import DataList

import torchvision.transforms as transforms
import numpy as np


class MulticlassDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """
    def __init__(self, opt, data_list):
        """
        Initialize this dataset class.
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
        transform_list = [transforms.ToTensor(), transforms.CenterCrop((512, 512))]
        trans = transforms.Compose(transform_list)

        labeled_slice = np.array(f['Dicom']).astype(np.float32) / 255.0
        width = labeled_slice.shape[1]
        mask = np.zeros([width, width, 4]).astype(np.float32)

        effective_channels = ''
        if 'None' in h5_path:
            effective_channels = '0123'
        label = np.zeros([width, width]).astype(np.float32)
        for key in f.keys():
            if key == 'ICAL':
                tmp_ical = np.array(f['ICAL']).astype(np.float32)
                label += 1 * (tmp_ical[..., 1] - tmp_ical[..., 0])
                mask[..., 0] = tmp_ical[..., 1] - tmp_ical[..., 0]
                effective_channels += '0'
            elif key == 'ICAR':
                tmp_icar = np.array(f['ICAR']).astype(np.float32)
                label += 2 * (tmp_icar[..., 1] - tmp_icar[..., 0])
                mask[..., 1] = tmp_icar[..., 1] - tmp_icar[..., 0]
                effective_channels += '1'
            elif key == 'ECAL':
                tmp_ecal = np.array(f['ECAL']).astype(np.float32)
                label += 3 * (tmp_ecal[..., 1] - tmp_ecal[..., 0])
                mask[..., 2] = tmp_ecal[..., 1] - tmp_ecal[..., 0]
                effective_channels += '2'
            elif key == 'ECAR':
                tmp_ecar = np.array(f['ECAR']).astype(np.float32)
                label += 4 * (tmp_ecar[..., 1] - tmp_ecar[..., 0])
                mask[..., 3] = tmp_ecar[..., 1] - tmp_ecar[..., 0]
                effective_channels += '3'

        labeled_slice = trans(labeled_slice)
        # tmp_mask = np.array(f['combing_mask']).astype(np.float32)
        # mask = tmp_mask[..., 1] - tmp_mask[..., 0]
        mask = trans(mask)
        label = trans(label.astype(np.longlong)).squeeze()
        return labeled_slice, mask, label, effective_channels

    def __len__(self):
        """
        Return the total number of images in the dataset.
        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return len(self.h5_list)
