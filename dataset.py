import os
import sys
import math
import numpy as np
import random
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from tqdm import tqdm

class SkeletonDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 label_path,
                 clip_size=150,
                 use_one_person=False,
                 normalization=False,
                 noise_addition=False,
                 noise_scale=1e-1,
                 ):
        self.data_path = data_path
        self.label_path = label_path
        self.clip_size = clip_size
        self.use_one_person = use_one_person
        self.normalization = normalization
        self.noise_addition = noise_addition
        self.noise_scale = noise_scale

        self.load_data()

    def load_data(self):
        '''
        skeleton shape

        1. batch_size
        2. dimensionality
        3. time_step
        4. joints
        5. people
        '''
        # load label
        if '.pkl' in self.label_path:
            try:
                with open(self.label_path) as f:
                    self.sample_name, self.label = pickle.load(f)
            except:
                # for pickle file from python2
                with open(self.label_path, 'rb') as f:
                    self.sample_name, self.label = pickle.load(
                        f, encoding='latin1')
        # old label format
        elif '.npy' in self.label_path:
            self.label = list(np.load(self.label_path))
            self.sample_name = [str(i) for i in range(len(self.label))]
        else:
            raise ValueError()

        # load data
        self.data = np.load(self.data_path)
        self.N, self.C, self.T, self.V, self.M = self.data.shape

        # self.data = self.data[:, :, 1:, :, :] - self.data[:, :, :-1, :, :]

        # order = [0,12,13,14,16,17,18,1,20,2,3,8,9,11,4,5,7]
        # self.data = self.data[:, :, :, order, :]
        #
        # origin = self.data[:, :, :, 7, :]
        # self.data = self.data - origin[:, :, :, None, :]

        # self.data[:, 0, :, :, :] = - self.data[:, 0, :, :, :]
        
        # l_spine = self.data[:, :, :, 8, :] - self.data[:, :, :, 0, :]
        # l_spine = np.linalg.norm(l_spine, ord=2, axis=1)
        # self.data = 0.5 * self.data / l_spine[:, None, :, None, :]
        #
        # self.data[np.isnan(self.data)] = 0
        # self.data[np.isinf(self.data)] = 0

        # normalization
        if self.normalization:
            self.data = self.data.transpose(0, 4, 2, 3, 1) # N M T V C
            # self.data = self.data - np.tile(self.data[:, :, :, 1:2, :],(1, 1, 1, self.V, 1))
            # self.data = self.data.reshape(self.N*self.M, self.T, self.V*self.C)
            # self.data = np.array([self.pararell_skeleton(n) for n in tqdm(self.data)])
            self.data = self.data.reshape(self.N*self.M*self.T, self.V*self.C)
            self.data = np.array([self.align_spine_length(i) for i in tqdm(self.data)])
            self.data = self.data.reshape(self.N, self.M, self.T, self.V, self.C).transpose(0, 4, 2, 3, 1)

        # noise asddition
        if self.noise_addition:
            self.data = self.data + np.random.normal(loc=0, scale=self.noise_scale, size=np.empty_like(self.data).shape)

        # clipping
        if self.use_one_person:
            self.data = self.data[:, :, :self.clip_size, :, 0:1]
        else:
            self.data = self.data[:, :, :self.clip_size, :, :]

        # save numpy
        # if self.data_path == './data/NTU-RGB-D/xsub/train_data.npy':
        #     np.save('./data/NTU-RGB-D/xsub/train_data_norm.npy', self.data)
        # elif self.data_path == './data/NTU-RGB-D/xsub/val_data.npy':
        #     np.save('./data/NTU-RGB-D/xsub/val_data_norm.npy', self.data)

    def y_transmat(self, thetas):
        tms = np.zeros((0, 3, 3))
        thetas = thetas*np.pi/180
        for theta in thetas:
            tm = np.zeros((3, 3))
            tm[0, 0] = np.cos(theta)
            tm[0, 2] = -np.sin(theta)
            tm[1, 1] = 1
            tm[2, 0] = np.sin(theta)
            tm[2, 2] = np.cos(theta)
            tm = tm[np.newaxis, :, :]
            tms = np.concatenate((tms, tm), axis=0)
        return tms

    def pararell_skeleton(self, raw_mat):
        '''
        raw_mat with the shape of (nframes, 25*3)
        '''
        joints_list = []

        for each_joints in range(25):
            joints_list.append(raw_mat[:, each_joints*3:each_joints*3+3])

        right_shoulder = joints_list[8] # 9th joint
        left_shoulder = joints_list[4] # 5tf joint
        vec = right_shoulder-left_shoulder
        vec[:, 1] = 0
        l2_norm = np.sqrt(np.sum(np.square(vec), axis=1))
        theta = vec[:, 0]/(l2_norm+0.0001)
        # print(l2_norm)
        thetas = np.arccos(theta)*(180/np.pi)
        isv = np.sum(vec[:, 2])
        if isv >= 0:
            thetas = -thetas
        y_tms = self.y_transmat(thetas)

        new_skel = np.zeros(shape=(0, 25*3))
        for ind, each_s in enumerate(raw_mat):
            r = np.reshape(each_s, newshape=(25, 3))
            r = np.transpose(r)
            r = np.dot(y_tms[ind], r)
            r_t = np.transpose(r)
            r_t = np.reshape(r_t, newshape=(1, -1))
            new_skel = np.concatenate((new_skel, r_t), axis=0)
        return new_skel

    def align_spine_length(self, raw_mat):
        '''
        raw_mat with the shape of (25*3)
        '''
        len_spine = np.sqrt(np.square(raw_mat[60]-raw_mat[0]) + np.square(raw_mat[61]-raw_mat[1]) \
                    + np.square(raw_mat[62]-raw_mat[2]))
        if len_spine != 0:
            new_skel = raw_mat * 0.5 / len_spine
        else:
            new_skel = raw_mat
        return new_skel

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        return data_numpy, label

def test(data_path, label_path, vid=None):
    import matplotlib.pyplot as plt
    loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, label_path),
        batch_size=64,
        shuffle=False,
        num_workers=2)

    if vid is not None:
        sample_name = loader.dataset.sample_name
        sample_id = [name.split('.')[0] for name in sample_name]
        index = sample_id.index(vid)
        data, label = loader.dataset[index]
        data = data.reshape((1, ) + data.shape)

        # for batch_idx, (data, label) in enumerate(loader):
        N, C, T, V, M = data.shape

        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)

        pose, = ax.plot(np.zeros(V * M), np.zeros(V * M), 'g^', marker='.', markeredgecolor='r')
        ax.axis([-1, 1, -1, 1])

        for n in range(N):
            for t in range(T):
                x = data[n, 0, t, :, 0]
                y = data[n, 1, t, :, 0]
                z = data[n, 2, t, :, 0]
                pose.set_xdata(x)
                pose.set_ydata(y)
                print('T: {}'.format(t))
                print(x)
                print(y)
                print(z)
                print(np.sqrt(np.square(x[20]-x[0]) + np.square(y[20]-y[0]) \
                            + np.square(z[20]-z[0])))
                fig.canvas.draw()
                plt.pause(1)

def visualize(befor_path, after_path):
    data_before = np.load(befor_path)
    data_after = np.load(after_path)

    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    C, T, V, M = data_before.shape

    pose1, = ax1.plot(np.zeros(V * M), np.zeros(V * M), 'g^', marker='.', markeredgecolor='r')
    pose2, = ax2.plot(np.zeros(V * M), np.zeros(V * M), 'g^', marker='.', markeredgecolor='r')
    ax1.axis([-1, 1, -1, 1])
    ax2.axis([-1, 1, -1, 1])

    for t in range(T):
        x1 = data_before[0, t, :, 0]
        y1 = data_before[1, t, :, 0]
        z1 = data_before[2, t, :, 0]
        x2 = data_after[0, t, :, 0]
        y2 = data_after[1, t, :, 0]
        z2 = data_after[2, t, :, 0]
        pose1.set_xdata(x1)
        pose1.set_ydata(y1)
        pose2.set_xdata(x2)
        pose2.set_ydata(y2)
        # print('T: {}'.format(t))
        # print(x)
        # print(y)
        # print(z)
        # print(np.sqrt(np.square(x[20]-x[0]) + np.square(y[20]-y[0]) \
        #             + np.square(z[20]-z[0])))
        fig.canvas.draw()
        plt.pause(0.5)


if __name__ == '__main__':
    data_path = "./data/NTU-RGB-D/xview/val_data.npy"
    label_path = "./data/NTU-RGB-D/xview/val_label.pkl"

    test(data_path, label_path, vid='S003C001P017R001A044')

    # epoch = '168'
    # before_path = './log/1reconst_mse/before_' + epoch + '.npy'
    # after_path = './log/1reconst_mse/after_' + epoch + '.npy'
    # visualize(before_path, after_path)
    # # epoch = '182'
    # # before_path = './log/1reconst_gn/before_' + epoch + '.npy'
    # # after_path = './log/1reconst_gn/after_' + epoch + '.npy'
    # # visualize(before_path, after_path)
