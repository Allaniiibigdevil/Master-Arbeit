import os

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader


class PhysioNet32(Dataset):
    def __init__(self, path, flag):
        super(PhysioNet32, self).__init__()
        self.path = path
        self.flag = flag
        self.data, self.labels = self.__read_data__()

    def __read_data__(self):
        assert self.flag in ['train', 'test']
        type_map = {'train': ['train_data_swm.csv', 'train_labels.csv'],
                    'test': ['test_data_swm.csv', 'test_labels.csv']}
        data_path = os.path.join(self.path, type_map[self.flag][0])
        labels_path = os.path.join(self.path, type_map[self.flag][1])
        data = pd.read_csv(data_path).values
        labels = pd.read_csv(labels_path).values
        return data, labels

    def __getitem__(self, index):
        data = self.data[32 * index: 32 * index + 32, :]
        label = self.labels[index][0]
        return data, label

    def __len__(self):
        data_len = self.labels.shape[0]
        return data_len


class PhysioNet640(Dataset):
    def __init__(self, path, flag):
        super(PhysioNet640, self).__init__()
        self.path = path
        self.flag = flag
        self.data, self.labels = self.__read_data__()

    def __read_data__(self):
        assert self.flag in ['train', 'test']
        type_map = {'train': ['train_data.csv', 'train_labels.csv'],
                    'test': ['test_data.csv', 'test_labels.csv']}
        data_path = os.path.join(self.path, type_map[self.flag][0])
        labels_path = os.path.join(self.path, type_map[self.flag][1])
        data = pd.read_csv(data_path).values
        labels = pd.read_csv(labels_path).values
        return data, labels

    def __getitem__(self, index):
        data = self.data[640 * index: 640 * index + 640, :]
        label = self.labels[index][0]
        return data, label

    def __len__(self):
        data_len = self.labels.shape[0]
        return data_len


class PhysioNetCV32(Dataset):
    def __init__(self, root_path, label_path, flag, cv):
        super(PhysioNetCV32, self).__init__()
        self.root_path = root_path
        self.label_path = label_path
        self.flag = flag
        self.cv = cv
        self.data, self.labels = self.__read_data__()

    def __read_data__(self):
        assert self.flag in ['train', 'test']
        type_map = {'train': ['train_data_cv1_swm.csv',
                              'train_data_cv2_swm.csv',
                              'train_data_cv3_swm.csv',
                              'train_data_cv4_swm.csv',
                              'train_data_cv5_swm.csv',
                              'train_labels_cv1.csv',
                              'train_labels_cv2.csv',
                              'train_labels_cv3.csv',
                              'train_labels_cv4.csv',
                              'train_labels_cv5.csv'],
                    'test': ['test_data_cv1_swm.csv',
                             'test_data_cv2_swm.csv',
                             'test_data_cv3_swm.csv',
                             'test_data_cv4_swm.csv',
                             'test_data_cv5_swm.csv',
                             'test_labels_cv1.csv',
                             'test_labels_cv2.csv',
                             'test_labels_cv3.csv',
                             'test_labels_cv4.csv',
                             'test_labels_cv5.csv']}
        data_path = os.path.join(self.root_path, type_map[self.flag][self.cv])
        labels_path = os.path.join(self.label_path, type_map[self.flag][self.cv + 5])
        data = pd.read_csv(data_path).values
        labels = pd.read_csv(labels_path).values
        return data, labels

    def __getitem__(self, index):
        data = self.data[32 * index: 32 * index + 32, :]
        label = self.labels[index][0]
        return data, label

    def __len__(self):
        data_len = self.labels.shape[0]
        return data_len


class PhysioNetCV640(Dataset):
    def __init__(self, root_path, label_path, flag, cv):
        super(PhysioNetCV640, self).__init__()
        self.root_path = root_path
        self.label_path = label_path
        self.flag = flag
        self.cv = cv
        self.data, self.labels = self.__read_data__()

    def __read_data__(self):
        assert self.flag in ['train', 'test']
        type_map = {'train': ['train_data_cv1.csv',
                              'train_data_cv2.csv',
                              'train_data_cv3.csv',
                              'train_data_cv4.csv',
                              'train_data_cv5.csv',
                              'train_labels_cv1.csv',
                              'train_labels_cv2.csv',
                              'train_labels_cv3.csv',
                              'train_labels_cv4.csv',
                              'train_labels_cv5.csv'],
                    'test': ['test_data_cv1.csv',
                             'test_data_cv2.csv',
                             'test_data_cv3.csv',
                             'test_data_cv4.csv',
                             'test_data_cv5.csv',
                             'test_labels_cv1.csv',
                             'test_labels_cv2.csv',
                             'test_labels_cv3.csv',
                             'test_labels_cv4.csv',
                             'test_labels_cv5.csv']}
        data_path = os.path.join(self.root_path, type_map[self.flag][self.cv])
        labels_path = os.path.join(self.label_path, type_map[self.flag][self.cv + 5])
        data = pd.read_csv(data_path).values
        labels = pd.read_csv(labels_path).values
        return data, labels

    def __getitem__(self, index):
        data = self.data[640 * index: 640 * index + 640, :]
        label = self.labels[index][0]
        return data, label

    def __len__(self):
        data_len = self.labels.shape[0]
        return data_len


class PhysioNetSC(Dataset):
    def __init__(self, path, flag):
        super(PhysioNetSC, self).__init__()
        self.path = path
        self.flag = flag
        self.data, self.labels = self.__read_data__()

    def __read_data__(self):
        assert self.flag in ['train', 'test']
        type_map = {'train': ['train_data.npy', 'train_labels.npy'],
                    'test': ['test_data.npy', 'test_labels.npy']}
        data_path = os.path.join(self.path, type_map[self.flag][0])
        labels_path = os.path.join(self.path, type_map[self.flag][1])
        data = np.load(data_path)
        labels = np.load(labels_path)
        return data, labels

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index][0]
        return data, label

    def __len__(self):
        data_len = self.labels.shape[0]
        return data_len


def data_provider(args, flag, cv):
    assert flag in ['train', 'test']
    # define batch_size and shuffle_flag for train loader and test loader
    if flag == 'train':
        batch_size = args.batch_size
        shuffle_flag = True
    else:
        batch_size = 1
        shuffle_flag = False
    # choose Dataset
    dataset_dict = {
        'PhysioNet32': PhysioNet32,
        'PhysioNet640': PhysioNet640,
        'PhysioNetCV32': PhysioNetCV32,
        'PhysioNetCV640': PhysioNetCV640,
        'PhysioNetSC': PhysioNetSC
    }
    # 旧数据集数据和标签在同文件夹下，无需cross validation
    if dataset_dict[args.data] in [PhysioNet32, PhysioNet640, PhysioNetSC]:
        dataset = dataset_dict[args.data](args.root_path, flag)
    # 新数据集数据和标签在不同文件夹下，需使用cross validation
    else:
        dataset = dataset_dict[args.data](args.root_path, args.label_path, flag, cv)
    print(flag, len(dataset))
    # define data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_flag)
    return dataset, data_loader
