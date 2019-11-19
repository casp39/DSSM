import argparse
import os
import shutil

import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

import numpy as np
from tqdm import tqdm

from model import *
from train import test_disc

parser = argparse.ArgumentParser(description='Deep State-Space Model for Action Recognition')

# dataset
parser.add_argument('--dataset_path', type=str, default='../data/NTU-RGB-D/')
parser.add_argument('--model_path', type=str, default='../log/test/model.pth')
parser.add_argument('--evaluation', type=str, default='xview', choices=['xview', 'xsub'])
parser.add_argument('--normalization', action='store_true')
parser.add_argument('--noise_addition', action='store_true')
parser.add_argument('--noise_scale', type=float, default=1e-1)
parser.add_argument('--num-worker', type=int, default=0)

# data
parser.add_argument('--time_step', type=int, default=150)
parser.add_argument('--x_dim', type=int, default=25*3*2)
parser.add_argument('--class_dim', type=int, default=60)

# batch size & epoch size
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=256)
parser.add_argument('--dssm_epochs', type=int, default=100)
parser.add_argument('--disc1_epochs', type=int, default=200)
parser.add_argument('--disc2_epochs', type=int, default=500)

# model
parser.add_argument('--infer_model', type=str, default='STLR', choices=['STLR', 'STR', 'SL', 'MFLR'])
parser.add_argument('--save_path', type=str, default='./log/test')

# hyper parameters
parser.add_argument('--beta', type=int, default=1)
parser.add_argument('--dropout_p', type=float, default=0.5)
parser.add_argument('--lr_dssm', type=float, default=1e-4)     #learning rate (feature extraction)
parser.add_argument('--lr_disc', type=float, default=1e-4)     #learning rate (discriminator)
parser.add_argument('--use_one_person', action='store_true')

# varialbe_size
parser.add_argument('--z_dim', type=int, default=50)               #z
parser.add_argument('--rnn_dim', type=int, default=800)            #h_left, h_right
parser.add_argument('--rnn_layer', type=int, default=2)
parser.add_argument('--z2comb_dim', type=int, default=400)         #z -> comb
parser.add_argument('--hcomb_dim', type=int, default=600)          #h_comb (equal to the hidden size of rnn)
parser.add_argument('--comb2z_dim', type=int, default=400)         #comb -> z
parser.add_argument('--emi_dim', type=int, default=200)            #z -> x
parser.add_argument('--trans_dim', type=int, default=400)          #z -> mu, logvar
parser.add_argument('--disc_dim', type=int, default=400)           #z -> y
parser.add_argument('--all_h_dim', type=int, default=512)

args = parser.parse_args()

def initialize():
    if os.path.isdir(args.save_path):
        shutil.rmtree(args.save_path)
    os.makedirs(args.save_path)
    if args.all_h_dim != 0:
        args.rnn_dim = args.z2comb_dim = args.hcomb_dim = args.comb2z_dim = \
        args.emi_dim = args.trans_dim = args.disc_dim = args.all_h_dim
    test_loader_kwargs ={
        'data_path':args.dataset_path + args.evaluation + '/val_data.npy',
        'label_path':args.dataset_path + args.evaluation + '/val_label.pkl',
        'clip_size':args.time_step,
        'use_one_person':args.use_one_person,
        'normalization':args.normalization,
        'noise_addition':args.noise_addition,
        'noise_scale':args.noise_scale}
    return test_loader_kwargs

def main():
    initialize()
    device = torch.device("cuda")

    test_loader_kwargs = initialize()

    param = torch.load(args.model_path)
    model = Net(args, device).to(device)
    model.load_state_dict(param)

    test_disc(args, model, device, test_loader_kwargs)

if __name__ == '__main__':
    main()
