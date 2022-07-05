### The Released Code for "Counterfactual Zero-shot and Open-Set Visual Recognition"
### Author: Wang Tan
### Part of Code borrow from "CGDL"

from __future__ import division
from email.headerregistry import ContentTransferEncodingHeader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import argparse
import os
import time
from utils import * 
from dataloader import MNIST_Dataset, CIFAR10_Dataset, SVHN_Dataset, CIFARAdd10_Dataset, CIFARAdd50_Dataset, CIFARAddN_Dataset
#from keras.utils import to_categorical
from model import LVAE, SupConLoss
from model2 import LVAE2
from qmv import ocr_test
from get_plots import *


# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch OSR Example')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 1e-3)')
    parser.add_argument('--wd', type=float, default=0.00, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.01, help='momentum (default: 1e-3)')
    parser.add_argument('--decreasing_lr', default='60,100,150', help='decreasing strategy')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='decreasing strategy')
    parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
    parser.add_argument('--seed_sampler', type=str, default='777 1234 2731 3925 5432', help='random seed for dataset sampler')
    parser.add_argument('--log_interval', type=int, default=20,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--val_interval', type=int, default=5, help='how many epochs to wait before another val')
    parser.add_argument('--test_interval', type=int, default=5, help='how many epochs to wait before another test')
    parser.add_argument('--lamda', type=int, default=100, help='lamda in loss function')
    parser.add_argument('--beta_z', type=int, default=1, help='beta of the kl in loss function')
    parser.add_argument('--beta_anneal', type=int, default=0, help='the anneal epoch of beta')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold of gaussian model')
    parser.add_argument('--tensorboard', action="store_true", default=False, help='If use tensorboard')
    parser.add_argument('--debug', action="store_true", default=False, help='If debug mode')

    # train
    parser.add_argument('--dataset', type=str, default="MNIST", help='The dataset going to use')
    parser.add_argument('--eval', action="store_true", default=False, help='directly eval?')
    parser.add_argument('--baseline', action="store_true", default=False, help='If is the bseline?')
    parser.add_argument('--usebb_model', action="store_true", default=False, help='If use model to get the train feature')
    parser.add_argument('--encode_z', type=int, default=None, help='If encode z and dim of z')
    parser.add_argument("--contrastive_loss", type=int, default=0, help="Use contrastive loss")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for contrastive loss")
    parser.add_argument("--contra_lambda", type=float, default=1.0, help="Scaling factor of contrastive loss")
    parser.add_argument("--save_epoch", type=int, default=None, help="save model in this epoch")
    parser.add_argument("--exp", type=int, default=0, help="which experiment")
    parser.add_argument("--unseen_num", type=int, default=13, help="unseen class num in CIFAR100")
        # mmd params
    parser.add_argument("--mmd_loss", type=int, default=0, help='Use MMD loss?')
    parser.add_argument('--s_jitter', type=float, default=0.5, help='Strength for color jitter')
    parser.add_argument('--p_grayscale', type=float, default=0.3, help='Probability of random grayscale')
    parser.add_argument('--gamma', type=float, default=1.0, help='Weight for kernel MMD loss')
    parser.add_argument('--eta', type=int, default=1, help='Weight for mmd loss')
    parser.add_argument('--kernel', type=str, default='multiscale', help='Which kernel to use for MMD loss')    
        # supcon params
    parser.add_argument('--supcon_loss', type=int, default=0, help='Use supcon loss? ')
    parser.add_argument('--con_temperature', type=float, default=0.07, help='Temperature for supcon loss')
    parser.add_argument('--theta', type=float, default=0.2, help='Weight for supcon loss')


    # test
    parser.add_argument('--cf', action="store_true", default=False, help='use counterfactual generation')
    parser.add_argument('--cf_threshold', action="store_true", default=False, help='use counterfactual threshold in revise_cf')
    parser.add_argument('--yh', action="store_true", default=False, help='use yh rather than feature_y_mean')
    parser.add_argument('--use_model_gau', action="store_true", default=False, help='use feature by model in gau')
    parser.add_argument('--correct_split', action='store_true', default=False, help='Use the correct test-val split?')
    parser.add_argument('--rm_skips', action='store_true', default=False, help='Remove skip connections? ')
    parser.add_argument('--no_aug', type=int, default=0, help="No augmentation for losses?")
    parser.add_argument('--get_plots', action= 'store_true', default=False, help='Plot images?')

    args = parser.parse_args()
    return args

def control_seed(args):
    # seed 
    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True


class DeterministicWarmup(object):
    def __init__(self, n=100, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.inc = 1 / n

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc

        self.t = self.t_max if t > self.t_max else t  # 0->1
        return self.t


if __name__ == '__main__':

    args = get_args()
    control_seed(args)

    if args.dataset == "MNIST":
        load_dataset = MNIST_Dataset(args.dataset)
        args.num_classes = 6
        in_channel = 1
    elif args.dataset == "CIFAR10":
        load_dataset = CIFAR10_Dataset(args.dataset)
        args.num_classes = 6
        in_channel = 3
    elif args.dataset == "SVHN":
        load_dataset = SVHN_Dataset(args.dataset)
        args.num_classes = 6
        in_channel = 3
    elif args.dataset == "CIFARAdd10":
        load_dataset = CIFARAdd10_Dataset(args.dataset)
        args.num_classes = 4
        in_channel = 3
    elif args.dataset == "CIFARAdd50":
        load_dataset = CIFARAdd50_Dataset(args.dataset)
        args.num_classes = 4
        in_channel = 3
    elif args.dataset == "TinyImageNet":
        load_dataset = TinyImageNet_Dataset(args.dataset)
        args.num_classes = 20
        in_channel = 3
    elif args.dataset == "CIFAR100":
        load_dataset = CIFAR100_Dataset(args.dataset)
        args.num_classes = 15
        in_channel = 3
    elif args.dataset == "CIFARAddN":
        load_dataset = CIFARAddN_Dataset(args.dataset, args.unseen_num)
        args.num_classes = 4
        in_channel = 3


    args.exp  = args.exp - 1
    exp_name = get_exp_name(args) # lamda100-...
    args.exp_name = exp_name
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    args.transforms = get_transforms(args.s_jitter, args.p_grayscale)
    config = vars(args)

    print("Experiment: {} \n with hyperparameters: {}".format(exp_name,config))

    ### run experiment 1/5 times
    for run_idx in range(args.exp, args.exp+1):
        print("Begin to Run Exp %s..." %run_idx)
        args.run_idx = run_idx
        seed_sampler = int(args.seed_sampler.split(' ')[run_idx])
        # seed_sampler = None
        save_path = 'results/%s' %(exp_name)
        args.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        latent_dim = 32
        if args.encode_z:
            latent_dim += args.encode_z

        if not args.rm_skips: 
            lvae = LVAE(in_ch=in_channel,
                    out_ch64=64, out_ch128=128, out_ch256=256, out_ch512=512,
                    kernel1=1, kernel2=2, kernel3=3, padding0=0, padding1=1, stride1=1, stride2=2,
                    flat_dim32=32, flat_dim16=16, flat_dim8=8, flat_dim4=4, flat_dim2=2, flat_dim1=1,
                    latent_dim512=512, latent_dim256=256, latent_dim128=128, latent_dim64=64, latent_dim32=latent_dim,
                    num_class=args.num_classes, dataset=args.dataset, args=args)
        else: 
            lvae = LVAE2(in_ch=in_channel,
                    out_ch64=64, out_ch128=128, out_ch256=256, out_ch512=512,
                    kernel1=1, kernel2=2, kernel3=3, padding0=0, padding1=1, stride1=1, stride2=2,
                    flat_dim32=32, flat_dim16=16, flat_dim8=8, flat_dim4=4, flat_dim2=2, flat_dim1=1,
                    latent_dim512=512, latent_dim256=256, latent_dim128=128, latent_dim64=64, latent_dim32=latent_dim,
                    num_class=args.num_classes, dataset=args.dataset, args=args)

        use_cuda = torch.cuda.is_available() and True
        device = torch.device("cuda" if use_cuda else "cpu")

        # data loader
        train_dataset, val_dataset, test_dataset = load_dataset.sampler(seed_sampler)

        if args.correct_split: 
            testdata_seen, testdata_unseen = test_dataset 
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=1, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=1, pin_memory=True)
            test_loader_seen = DataLoader(testdata_seen, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=1, pin_memory=True)
            test_loader_unseen = DataLoader(testdata_unseen, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=1, pin_memory=True)
        else: 
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
            test_loader_seen = val_loader 
            test_loader_unseen = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

        # Model
        lvae.to(args.device)
        nllloss = nn.NLLLoss().to(device)

        # optimzer
        optimizer = optim.SGD(lvae.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
        print('decreasing_lr: ' + str(decreasing_lr))
        beta = DeterministicWarmup(n=50, t_max=1)  # Linear warm-up from 0 to 1 over 50 epoch
        if args.beta_anneal != 0:
            args.beta_anneal = DeterministicWarmup(n=args.beta_anneal, t_max=args.beta_z)
        lvae.supcon_critic = SupConLoss(args.con_temperature)


            # load train model
        states = torch.load(os.path.join(args.save_path, 'model.pkl'), map_location=args.device)
        lvae.load_state_dict(states['model'])

        plot_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, drop_last=False, num_workers=1, pin_memory=True)
        plot_rec(lvae, plot_loader, args.transforms, args, mean=load_dataset.mean, std=load_dataset.std)

        point_cloud(lvae, dataset)

        
        ocr_test(args, lvae, train_loader, test_loader_seen, test_loader_unseen)