import os
import h5py
import matplotlib.pyplot as plt

cdir = '../dataset/careIIChallenge/normalized/'

files = os.listdir(cdir)
init = 0
for file in files:
    if 'None' in file:
        continue
    f = h5py.File(cdir + file, 'r')
    plt.figure(figsize=(10, 10))
    plt.imshow(f['Dicom'])
    plt.show()
    plt.imshow(f['combing_mask'][..., 0])
    plt.show()



