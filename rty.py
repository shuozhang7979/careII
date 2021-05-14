import torch
import numpy as np
import h5py
from torch.autograd import Variable
from torchvision.transforms import transforms
import torch.nn.functional as f
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from data.k_folder import DataList
import time


# if __name__ == '__main__':
#     opt = TrainOptions().parse()
#     # get training options
#     data_list = DataList(opt.dataset_root)
#
#     val_list = data_list.get_current_list(phase='val', k=opt.k_fold, time=opt.time)
#     val_dataset = create_dataset(opt, val_list)
#     val_data = val_dataset.__iter__()
#     # for i in val_dataset:
#     #     print(i[3])
#     for i in range(100):
#         k = next(val_data)
#         print(k[3])
# fake = Variable(torch.randn(2, 3, 4, 4))
# print(fake)
# mask = Variable(torch.LongTensor(2, 4, 4).random_(3))
# # fake = f.softmax(fake, dim=1)
#
# print(mask)
# print(fake.shape, mask.shape)
# sum = 0
# for i in range(3):
#     sum += fake[0, i, 3, 3]
# print(sum)
# loss = f.cross_entropy(fake, mask)
# print(loss)

# list_a = [i for j in range(10) for i in range(5)]
#
# print(list_a[0:2])
# lis1 = np.array([2, 3, 4, 5])
# lis2 = np.array([1, 2, 3, 0])
# list_1 = [0 for _ in range(20)]
# list_2 = [1 for _ in range(3)]
# print(list_1, list_2)
# name = '..\\dataset\\careIIChallenge\\normalized\\P530_346_ICARECAL.h5'
# if 'ECAL' in name:
#     print('dad')
# toral_list = np.stack((list))
# print(lis1[lis2])
# path = 'dataset/careIIChallenge/normalized/P125_346_ICALICAR.h5'
# with h5py.File(path, 'r') as f:
#     print(f.keys())
#
#     h = np.array(f['ICA']).astype(np.float32)
#
# print(h)
# array_1 = np.random.randn(4, 4, 5)
# label = np.random.randint(0, 5, [4, 4])
# print(array_1)
# print(label)
# transform_list = [transforms.ToTensor()]
# trans = transforms.Compose(transform_list)
# t = trans(array_1).unsqueeze(0)
# t = torch.cat((t, t))
# l = trans(label.astype(np.longlong)).squeeze().unsqueeze(0)
# l = torch.cat((l, l))
# print(t.shape, l.shape)
# class_loss = torch.nn.CrossEntropyLoss()
# loss = class_loss(t, l)
# print(loss)

# visual_names = ['one_slice_v', 'masks_v', 'fake_masks_v',
#                 ['masks_v_1', 'masks_v_0', 'fake_masks_v_1', 'fake_masks_v_0']]
#
# one = torch.ones([2, 1, 512, 512])
# print(one.shape)
# one = one.squeeze(0)
# print(one.shape)


# x = torch.ones(2, 2, requires_grad=True)
# a = torch.ones(2, 2, requires_grad=True)
# y = x + 2
# z = y * y * 3
# z = z - a
# out = z.mean()
# out.backward()
#
# print(y.retain_grad)

# x = torch.randint(0, 100, (1, 9, 10, 10)) / 1.0
# y = torch.softmax(x, dim=1)
# z = torch.max(y, dim=1)
# print(z.indices)
# res = torch.zeros_like(x)
# for i in range(9):
#     res[:, i, :, :] = (z.indices == i).int()
# print(torch.sum(res, dim=1))


# def decreace_by_constant(list_a, start, end):
#     if (end - start) == 1:
#         pos = start
#         value = list_a[pos]
#         return pos, value
#     pos, value = decreace_by_constant(list_a, start+1, end)
#     if list_a[start] < value:
#         pos = start
#         value = list_a[pos]
#     return pos, value
#
#
# list_test = [5, 4, 3, 7]
# p, v = decreace_by_constant(list_test, 0, len(list_test))
# print('pos = %d, value = %d' % (p, v))

# list_test = [[5, 4, 3, 7], [2, 3]]
# dices_bs = torch.zeros(np.array(list_test).shape)
#
# print(dices_bs)
# effective_channels = []
# inf = ('01234', '531276')
# for i, str_i in enumerate(inf):
#     effective_channels.append([])
#     for j in str_i:
#         effective_channels[i].append(int(j))
# print(effective_channels)
# time_str = time.localtime()
# time_str = str(time_str.tm_year) + '_' + str(time_str.tm_mon) + '_' + str(time_str.tm_mday)
# print(time_str)

x = torch.ones(1, 2, 100, 100)
model = torch.nn.Conv2d(2, 3, 3)
y = model(x)
print(y.shape)
# for name, para in model.named_parameters():
#     print(para.data.shape)



