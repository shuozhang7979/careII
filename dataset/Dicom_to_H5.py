import os
import pydicom
import glob
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


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


def getContour(qvsroot, dicomslicei, conttype, dcmsz=720):
    qvasimg = qvsroot.findall('QVAS_Image')
    # if dicomslicei - 1 > len(qvasimg):
    #     print('no slice', dicomslicei)
    #     return
    # print( int(qvasimg[dicomslicei - 1].get('ImageName').split('I')[-1]) , dicomslicei)
    # assert int(qvasimg[dicomslicei - 1].get('ImageName').split('I')[-1]) == dicomslicei

    conts = qvasimg[dicomslicei - 1].findall('QVAS_Contour')
    tconti = -1
    for conti in range(len(conts)):
        if conts[conti].find('ContourType').text == conttype:
            tconti = conti
            break
    if tconti == -1:
        # print('no such contour', conttype)
        return
    pts = conts[tconti].find('Contour_Point').findall('Point')
    contours = []
    for pti in pts:
        contx = float(pti.get('x')) / 512 * dcmsz
        conty = float(pti.get('y')) / 512 * dcmsz
        # from space 512*512 to 720*720
        # if current pt is different from last pt, add to contours
        if len(contours) == 0 or contours[-1][0] != contx or contours[-1][1] != conty:
            contours.append([contx, conty])
    return np.array(contours)


def coordinate_to_mask(contour, mask_size):
    """
    :param mask_size:
    :param contour: a 2d list
    :return: a mask 720*720
    """
    mask_ = np.zeros((mask_size, mask_size))
    cv2.polylines(mask_, np.int32([contour]), 1, 1)
    # cv2.drawContours
    # cv2.fillConvexPoly
    cv2.fillPoly(mask_, np.int32([contour]), 1)
    return mask_


class KFold:
    def __init__(self, train, val, test):
        self.train = train
        self.val = val
        self.test = test
        self.num = 0

    def get_phase(self):
        self.num += 1
        if self.num <= self.train:
            return 'train'
        if self.num <= self.train + self.val:
            return 'val'
        if self.num <= self.train + self.val + self.test:
            return 'test'
        self.num = 0
        return self.get_phase()


cdir = 'E:\\datasets\\careIIChallenge'

dirs = ['../dataset/careIIChallenge/normalized/']
for dir_ in dirs:
    if not os.path.exists(dir_):
        os.makedirs(dir_)

"""
one slice one h5
"""

#

# kfold = KFold(5, 1, 1)
# for casei in os.listdir(cdir):
#     if casei[0] != '0':
#         continue
#     if casei[3:6] == '801':
#         continue
#     if casei[3:6] == '556':
#         continue
#     if casei[3:6] == '887':
#         continue
#     pi = casei.split('_')[1]
#     dcm_img = readDicom(cdir + '/' + casei)
#     for arti in ['ICAL', 'ICAR', 'ECAL', 'ECAR']:
#         cas_dir = cdir + '/' + casei + '/CASCADE-' + arti
#         qvs_path = cas_dir + '/E' + pi + 'S101_L.QVS'
#         qvsroot = ET.parse(qvs_path).getroot()
#         avail_slice = listContourSlices(qvsroot)
#         for i in range(len(avail_slice)):
#             dicom_slice_i = avail_slice[i]
#             lumen_cont = getContour(qvsroot, dicom_slice_i, 'Lumen', dcmsz=dcm_img.shape[0])
#             wall_cont = getContour(qvsroot, dicom_slice_i, 'Outer Wall', dcmsz=dcm_img.shape[0])
#             lumen_mask = coordinate_to_mask(lumen_cont.tolist(), dcm_img.shape[0])
#             wall_mask = coordinate_to_mask(wall_cont.tolist(), dcm_img.shape[0])
#             one_dcm = dcm_img[:, :, dicom_slice_i]
#             # normalize
#             mu = np.mean(one_dcm)
#             sigma = np.std(one_dcm)
#             one_dcm = (one_dcm - mu) / sigma
#
#             one_mask = np.stack((lumen_mask, wall_mask), axis=2)
#             # phase = kfold.get_phase()
#             print('../dataset/careIIChallenge/normalized/%s_%s_%d.h5' % (pi, arti, dicom_slice_i))
#             with h5py.File('../dataset/careIIChallenge/normalized/%s_%s_%d.h5' % (pi, arti, dicom_slice_i), 'w') as f:
#                 f['Dicom'] = one_dcm
#                 f['mask'] = one_mask

#     # plt.figure(figsize=(10, 40))
#     # plt.subplot(3, 1, 1)
#     # plt.imshow(one_dcm+lumen_mask*230)
#     #
#     # plt.subplot(3, 1, 2)
#     # plt.imshow(one_dcm)
#     # lumen_cont_mask = mask_to_contour(lumen_mask)
#     # plt.plot(lumen_cont_mask[:, 0], lumen_cont_mask[:, 1], 'bo', markersize=1)
#     #
#     # plt.subplot(3, 1, 3)
#     # plt.imshow(dcm_img[:, :, dicom_slice_i])
#     # plt.plot(lumen_cont[:, 0], lumen_cont[:, 1], 'ro', markersize=1)
#     #
#     # plt.show()


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
    dcm_img = readDicom(cdir + '/' + casei)
    for i in range(dcm_img.shape[2]):
        one_dcm = dcm_img[:, :, i]
        dis = {}
        for arti in ['ICAL', 'ICAR', 'ECAL', 'ECAR']:
            cas_dir = cdir + '/' + casei + '/CASCADE-' + arti
            qvs_path = cas_dir + '/E' + pi + 'S101_L.QVS'
            qvsroot = ET.parse(qvs_path).getroot()

            lumen_cont = getContour(qvsroot, i, 'Lumen', dcmsz=dcm_img.shape[0])
            if lumen_cont is not None:
                wall_cont = getContour(qvsroot, i, 'Outer Wall', dcmsz=dcm_img.shape[0])
                lumen_mask = coordinate_to_mask(lumen_cont.tolist(), dcm_img.shape[0])
                wall_mask = coordinate_to_mask(wall_cont.tolist(), dcm_img.shape[0])
                one_mask = np.stack((lumen_mask, wall_mask), axis=2)
                dis[arti] = one_mask

        combine_mask = np.zeros((dcm_img.shape[0], dcm_img.shape[0], 2))
        inf = ''
        for key, value in dis.items():
            inf += key
        if inf == '':
            inf = 'None'
        with h5py.File('../dataset/careIIChallenge/normalized/%s_%d_%s.h5' % (pi, i, inf), 'w') as f:
            f['Dicom'] = one_dcm
            for key, value in dis.items():
                f[key] = value
                combine_mask += value
            f['combing_mask'] = combine_mask


