import os
import pydicom
import glob
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from util.mask_to_contour import *


def readDicom(path):
    """
    :param path: to a case dir
    :return: numpy float64
    """
    pi = os.path.basename(path).split('_')[1]
    # case id
    dcm_size = len(glob.glob(path + '/*.dcm'))
    # read all files under path which match '*.dcm'
    dcms = [path + '/E' + pi + 'S101I%d.dcm' % dicom_slicei for dicom_slicei in range(1, dcm_size + 1)]
    # a path list of .dcm file path, total 720
    dcm_f = pydicom.read_file(dcms[0]).pixel_array
    # one dcm np (100, 720)
    dcm_size = max(dcm_f.shape)
    dcm_img = np.zeros((dcm_size, dcm_size, len(dcms)))
    for dcmi in range(len(dcms)):
        cdcm = pydicom.read_file(dcms[dcmi]).pixel_array
        # each file to numpy
        dcm_img[dcm_size // 2 - cdcm.shape[0] // 2:dcm_size // 2 + cdcm.shape[0] // 2,
        dcm_size // 2 - cdcm.shape[1] // 2:dcm_size // 2 + cdcm.shape[1] // 2, dcmi] = cdcm
        # each file: from (100,720) extend to the center of (720, 720), padding 0
    return dcm_img


def listContourSlices(qvsroot):
    avail_slices = []
    qvasimg = qvsroot.findall('QVAS_Image')
    # findall match 'QVAS_Image'
    for dicom_slicei in range(dcm_img.shape[2]):
        conts = qvasimg[dicom_slicei - 1].findall('QVAS_Contour')
        # for each qvasimg find contour
        if len(conts):
            avail_slices.append(dicom_slicei)
    return avail_slices


cdir = 'E:\\datasets\\careIIChallenge'
dir_slice ='E:\datasets\careIIChallenge\\normalized'

max_dict, min_dict = {}, {}
for casei in os.listdir(cdir):
    if casei[0] != '0':
        continue
    if casei[3:6] == '801':
        continue
    if casei[3:6] == '556':
        continue
    if casei[3:6] == '887':
        continue
    pi = casei.split('_')[1]
    max_slice = -1
    min_slice = 1000
    dcm_img = readDicom(cdir + '/' + casei)
    # np float64

    for arti in ['ICAL', 'ICAR', 'ECAL', 'ECAR']:
        # for four cascade
        cas_dir = cdir + '/' + casei + '/CASCADE-' + arti
        # cascade dir
        qvs_path = cas_dir + '/E' + pi + 'S101_L.QVS'
        # qvs file(xml)
        qvsroot = ET.parse(qvs_path).getroot()
        # xml root
        avail_slices = listContourSlices(qvsroot)
        for i in avail_slices:
            if i < min_slice:
                min_slice = i
            if i > max_slice:
                max_slice = i

    max_dict[pi] = max_slice
    min_dict[pi] = min_slice
    print(pi, max_slice, min_slice)


def is_fake_none(name):
    if 'None' in name:
        pi = name.split('_')[0]
        slices = int(name.split('_')[1])
        if min_dict[pi] <= slices <= max_dict[pi] or (max_dict[pi] == -1 or min_dict[pi] == 1000):
            print('%d %d %d remove %s' % (min_dict[pi], slices, max_dict[pi], name))
            return True
    return False


for i in os.listdir(dir_slice):
    if is_fake_none(i):
        os.remove(dir_slice+'\\'+i)
    else:
        pass
