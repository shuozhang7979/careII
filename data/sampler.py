from torch.utils.data import Sampler
import os
import numpy as np
import torch


class BalanceSampler(Sampler):
    #   method 2 sampler
    def __init__(self, h5_list):
        self.h5_list = h5_list
        self.NoneList = []
        self.ICARList = []
        self.ECARList = []
        self.ICALList = []
        self.ECALList = []
        for i, data in enumerate(self.h5_list):
            if 'None' in data:
                self.NoneList.append(i)
            if 'ICAR' in data:
                self.ICARList.append(i)
            if 'ECAR' in data:
                self.ECARList.append(i)
            if 'ICAL' in data:
                self.ICALList.append(i)
            if 'ECAL' in data:
                self.ECALList.append(i)
        self.max_sub_len = max(len(self.ECALList), len(self.ICALList),
                               len(self.ICARList), len(self.ICALList))
        self.NoneList = np.array(self.NoneList)
        self.ECALList = np.array(self.ECALList)
        self.ICALList = np.array(self.ICALList)
        self.ECARList = np.array(self.ECARList)
        self.ICARList = np.array(self.ICARList)

        print(len(self.NoneList), len(self.ECALList), len(self.ICALList),
              len(self.ICARList), len(self.ICALList))

        idx_icar = np.random.choice(self.ICARList.shape[0], self.max_sub_len, replace=True)
        idx_ecar = np.random.choice(self.ECARList.shape[0], self.max_sub_len, replace=True)
        idx_ical = np.random.choice(self.ICALList.shape[0], self.max_sub_len, replace=True)
        idx_ecal = np.random.choice(self.ECALList.shape[0], self.max_sub_len, replace=True)
        idx_none = np.random.choice(self.NoneList.shape[0], self.max_sub_len, replace=True)
        self.NoneList = self.NoneList[idx_none]
        self.ECALList = self.ECALList[idx_ecal]
        self.ICALList = self.ICALList[idx_ical]
        self.ECARList = self.ECARList[idx_ecar]
        self.ICARList = self.ICARList[idx_icar]
        self.TotalList = np.stack((self.NoneList, self.ECARList, self.ICARList, self.ICALList, self.ECALList))

    def __iter__(self):
        return iter([self.TotalList[i][j] for j in range(self.max_sub_len) for i in range(5)])

    def __len__(self):
        return self.max_sub_len * 5


# class BalanceInfSampler(Sampler):
#     #   method 1 sampler
#     def __init__(self, h5_list):
#         self.h5_list = h5_list
#         self.NoneList = []
#         self.ICARList = []
#         self.ECARList = []
#         self.ICALList = []
#         self.ECALList = []
#         for i, data in enumerate(self.h5_list):
#             if 'None' in data:
#                 self.NoneList.append(i)
#             if 'ICAR' in data:
#                 self.ICARList.append(i)
#             if 'ECAR' in data:
#                 self.ECARList.append(i)
#             if 'ICAL' in data:
#                 self.ICALList.append(i)
#             if 'ECAL' in data:
#                 self.ECALList.append(i)
#         self.max_sub_len = max(len(self.ECALList), len(self.ICALList),
#                                len(self.ICARList), len(self.ICALList))
#         self.NoneList = np.array(self.NoneList)
#         self.ECALList = np.array(self.ECALList)
#         self.ICALList = np.array(self.ICALList)
#         self.ECARList = np.array(self.ECARList)
#         self.ICARList = np.array(self.ICARList)
#
#         idx_icar = np.random.choice(self.ICARList.shape[0], self.max_sub_len, replace=True)
#         idx_ecar = np.random.choice(self.ECARList.shape[0], self.max_sub_len, replace=True)
#         idx_ical = np.random.choice(self.ICALList.shape[0], self.max_sub_len, replace=True)
#         idx_ecal = np.random.choice(self.ECALList.shape[0], self.max_sub_len, replace=True)
#         idx_none = np.random.choice(self.NoneList.shape[0], self.max_sub_len, replace=True)
#         self.NoneList = self.NoneList[idx_none]
#         self.ECALList = self.ECALList[idx_ecal]
#         self.ICALList = self.ICALList[idx_ical]
#         self.ECARList = self.ECARList[idx_ecar]
#         self.ICARList = self.ICARList[idx_icar]
#         self.TotalList = np.stack((self.NoneList, self.ECARList, self.ICARList, self.ICALList, self.ECALList))
#
#     def __iter__(self):
#         return iter([(self.TotalList[i][j], i) for j in range(self.max_sub_len) for i in range(5)])
#
#     def __len__(self):
#         return self.max_sub_len * 5
