import os


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF', '.h5'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def sort_balance(images):
    none_list = []
    icar_list = []
    ecar_list = []
    ical_list = []
    ecal_list = []
    for _, data in enumerate(images):
        if 'None' in data:
            none_list.append(data)
        if 'ICAR' in data:
            icar_list.append(data)
        if 'ECAR' in data:
            ecar_list.append(data)
        if 'ICAL' in data:
            ical_list.append(data)
        if 'ECAL' in data:
            ecal_list.append(data)

    len_none = len(none_list)
    len_icar = len(icar_list)
    len_ecar = len(ecar_list)
    len_ical = len(ical_list)
    len_ecal = len(ecal_list)
    min_len = min(len_none, len_icar, len_ecar, len_ical, len_ecal)
    p_none = len_none // min_len
    p_icar = len_icar // min_len
    p_ecar = len_ecar // min_len
    p_ical = len_ical // min_len
    p_ecal = len_ecal // min_len

    res = []
    for i in range(min_len):
        for j in range(p_none):
            res.append(none_list[i * p_none + j])
        for j in range(p_icar):
            res.append(icar_list[i * p_icar + j])
        for j in range(p_ical):
            res.append(ical_list[i * p_ical + j])
        for j in range(p_ecar):
            res.append(ecar_list[i * p_ecar + j])
        for j in range(p_ecal):
            res.append(ecal_list[i * p_ecal + j])
    return res


class DataList:
    def __init__(self, dirc, max_dataset_size=float("inf")):
        self.images = []
        # dir += ''
        assert os.path.isdir(dirc), '%s is not a valid directory' % dirc
        for root, _, fnames in sorted(os.walk(dirc)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    self.images.append(path)
        self.images = sort_balance(self.images)

    def get_current_list(self, phase='train', k=5, time=0):     # time [1, k-1]
        assert k >= 3
        slice_len = len(self.images) // k
        test_list = self.images[k-1*slice_len:]
        val_list = self.images[time*slice_len:(time+1)*slice_len]
        train_list = self.images[0: time*slice_len] + self.images[(time+1)*slice_len: k-1*slice_len]
        if phase == 'train':
            return train_list
        if phase == 'val':
            return val_list
        if phase == 'test':
            return test_list
        if phase == 'total':
            return self.images

