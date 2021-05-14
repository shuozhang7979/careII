import os
import pydicom
import glob
import cv2
import numpy as np
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
    # return dcm_img
    # dcm_img = np.zeros((dcm_f.shape[0], dcm_f.shape[1], len(dcms)))
    # for dcmi in range(len(dcms)):
    #     cdcm = pydicom.read_file(dcms[dcmi]).pixel_array
    #     dcm_img[:, :, dcmi] = cdcm
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
    if dicomslicei - 1 > len(qvasimg):
        print('no slice', dicomslicei)
        return
    assert int(qvasimg[dicomslicei - 1].get('ImageName').split('I')[-1]) == dicomslicei
    conts = qvasimg[dicomslicei - 1].findall('QVAS_Contour')
    tconti = -1
    for conti in range(len(conts)):
        if conts[conti].find('ContourType').text == conttype:
            tconti = conti
            break
    if tconti == -1:
        print('no such contour', conttype)
        return
    pts = conts[tconti].find('Contour_Point').findall('Point')
    contours = []
    for pti in pts:
        contx = float(pti.get('x')) / 512 * dcmsz
        conty = float(pti.get('y')) / 512 * dcmsz
        # from space 512*512 to 720*720
        #if current pt is different from last pt, add to contours
        if len(contours) == 0 or contours[-1][0] != contx or contours[-1][1] != conty:
            contours.append([contx, conty])
    return np.array(contours)


def coordinate_to_mask(contour):
    """
    :param contour: a 2d list
    :return: a mask
    """
    mask_ = np.zeros((720, 720))
    cv2.polylines(mask_, np.int32([contour]), 1, 1)
    # cv2.drawContours
    # cv2.fillConvexPoly
    cv2.fillPoly(mask_, np.int32([contour]), 1)
    return mask_


cdir = '../careIIChallenge'

for casei in os.listdir(cdir)[:1]:

    pi = casei.split('_')[1]

    dcm_img = readDicom(cdir + '/' + casei)
    # np float64
    print('Dcm shape', dcm_img.shape)
    for arti in ['ICAL', 'ICAR', 'ECAL', 'ECAR']:
        # for four cascade
        cas_dir = cdir + '/' + casei + '/CASCADE-' + arti
        # cascade dir
        qvs_path = cas_dir + '/E' + pi + 'S101_L.QVS'
        # qvs file(xml)
        qvsroot = ET.parse(qvs_path).getroot()
        # xml root
        avail_slices = listContourSlices(qvsroot)
        print('case', pi, 'art', arti, 'avail_slices', avail_slices)
        #
        if len(avail_slices):
            dicom_slicei = avail_slices[0]
            # first contour slice
            print('Displaying the contours for the first slice for', arti)
            lumen_cont = getContour(qvsroot, dicom_slicei, 'Lumen')
            # numpy (30, 2) points
            wall_cont = getContour(qvsroot, dicom_slicei, 'Outer Wall')
            plt.figure(figsize=(10, 10))
            plt.imshow(dcm_img[:, :, dicom_slicei])
            plt.plot(lumen_cont[:, 0], lumen_cont[:, 1], 'ro', markersize=1)
            # plt.plot(wall_cont[:, 0], wall_cont[:, 1], 'bo', markersize=1)
            plt.show()

