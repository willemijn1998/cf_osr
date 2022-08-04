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
# from utils_plot import get_tsne, scatter_plot, point_cloud, rec_histogram, get_sample_feas, get_class_means
from utils_plot import * 


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
    parser.add_argument('--use_model', action="store_true", default=False, help='If use model to get the train feature')
    parser.add_argument('--encode_z', type=int, default=None, help='If encode z and dim of z')
    parser.add_argument("--contrastive_loss", type=int, default=0, help="Use contrastive loss")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for contrastive loss")
    parser.add_argument("--contra_lambda", type=float, default=1.0, help="Scaling factor of contrastive loss")
    parser.add_argument("--rec_lamda", type=float, default=1.0, help="Scaling factor of reconstruction loss")
    parser.add_argument("--save_epoch", type=int, default=None, help="save model in this epoch")
    parser.add_argument("--exp", type=int, default=1, help="which experiment")
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
    parser.add_argument('--rm_skips', type=int, default=0, help='Remove skip connections? ')
    parser.add_argument('--no_aug', type=int, default=0, help="No augmentation for losses?")

    #plots
    parser.add_argument('--scatter', action= 'store_true', default=False, help='Generate scatter image?')
    parser.add_argument('--pointcloud', action= 'store_true', default=False, help='Pointcloud plot?')
    parser.add_argument('--rec_histo', action= 'store_true', default=False, help='Histogram of Reconstruction errors?')    
    parser.add_argument('--sample_scatter', action= 'store_true', default=False, help='Scatter plot of sample feas?')
    parser.add_argument('--ll_hist', action= 'store_true', default=False, help='Loglikelihood hist of class feats? ')
    parser.add_argument('--ll_hist_1c', action= 'store_true', default=False, help='Loglikelihood hist of one class feats? ')
    parser.add_argument('--scatter_layers', action= 'store_true', default=False, help='scatter plots of layers? ')
    parser.add_argument('--reconstruct', action= 'store_true', default=False, help='Show reconstructed imgs?')



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

        use_cuda = torch.cuda.is_available() and True
        device = torch.device("cuda" if use_cuda else "cpu")

        # data loader
        train_dataset, val_dataset, test_dataset = load_dataset.sampler(seed_sampler)

        if args.correct_split: 
            testdata_seen, testdata_unseen = test_dataset 
        else: 
            testdata_seen = val_dataset 

        if args.scatter: 
            # Scatter plot of the content features 
            train_feas = np.loadtxt((save_path + "/train_fea.txt"))
            train_tars = np.loadtxt((save_path + "/train_tar.txt"))            
            plot_name = exp_name[:-2] + exp_name[-1]
            feas = np.loadtxt((save_path + "/test_fea.txt"))
            targets = np.loadtxt((save_path + "/test_tar.txt"))

            means= get_class_means(train_feas, train_tars, args.num_classes)

            feas = np.vstack([feas, means])
            projections = get_tsne(feas, 2)

            means = projections[-args.num_classes:,:]
            projections = projections[:-args.num_classes,:]
            scatter_plot(projections, targets, plot_name, means)

        if args.pointcloud: 
            # Point cloud of the content features 
            plot_name = exp_name[:-2] + exp_name[-1]
            feas = np.loadtxt((save_path + "/test_fea.txt"))
            targets = np.loadtxt((save_path + "/test_tar.txt"))           
            projections = get_tsne(feas, 3)
            point_cloud(projections, targets, plot_name)
        
        if args.rec_histo: 
            # Histogram of the reconstruction losses 
            rec_path = args.save_path + "/test_rec.txt"
            recs = np.loadtxt(rec_path)
            tar_path = args.save_path + "/test_tar.txt"
            tars= np.loadtxt(tar_path)
            rec_histogram(recs, tars, args.num_classes, args.exp_name)

        if args.sample_scatter: 
            # Scatter plot of the sample/ style features 
            lvae = LVAE(in_ch=in_channel,
                    out_ch64=64, out_ch128=128, out_ch256=256, out_ch512=512,
                    kernel1=1, kernel2=2, kernel3=3, padding0=0, padding1=1, stride1=1, stride2=2,
                    flat_dim32=32, flat_dim16=16, flat_dim8=8, flat_dim4=4, flat_dim2=2, flat_dim1=1,
                    latent_dim512=512, latent_dim256=256, latent_dim128=128, latent_dim64=64, latent_dim32=latent_dim,
                    num_class=args.num_classes, dataset=args.dataset, args=args)
            states = torch.load(os.path.join(args.save_path, 'model.pkl'), map_location=args.device)
            lvae.load_state_dict(states['model'])
            
            test_loader_seen = DataLoader(testdata_seen, batch_size=args.batch_size, shuffle=False, drop_last=False)
            test_loader_unseen = DataLoader(testdata_unseen, batch_size=args.batch_size, shuffle=False, drop_last=False)
            sample_feas, sample_tars = get_sample_feas(test_loader_seen, test_loader_unseen, lvae, args)
            projections = get_tsne(sample_feas, 2)
            plot_name = exp_name[:-2] + exp_name[-1] + "_sample"
            scatter_plot(projections, sample_tars, plot_name)

    
        if args.ll_hist: 
            # Histogram of max-loglikelihoods for knowns and unknowns 
            testfea = np.loadtxt((args.save_path + "/test_fea.txt"))
            testtar = np.loadtxt((args.save_path + "/test_tar.txt"))     
            mu, sigma = get_mu_sigma(testtar, testfea, args.num_classes)
            ll_known, ll_unknown = ll_histogram(testtar, testfea, args.num_classes, mu, sigma)
            b1 = np.max(ll_known)
            bins = np.linspace(0, int(b1), 100)

            plt.hist(ll_known, bins, alpha=0.5, label='seen')
            plt.hist(ll_unknown, bins, alpha=0.5, label='unseen')
            plt.legend()
            plt.title(exp_name)
            plot_path = exp_name[:-2] + exp_name[-1]
            plt.savefig('images/ll_hist/%s.png' %(plot_path))
            plt.show()


        if args.ll_hist_1c: 
            # Histogram of the loglikelihoods per class knowns vs unknowns 
            testfea = np.loadtxt((args.save_path + "/test_fea.txt"))
            testtar = np.loadtxt((args.save_path + "/test_tar.txt"))     
            mu, sigma = get_mu_sigma(testtar, testfea, args.num_classes)


        if args.scatter_layers: 
            # Scatter plot of all L latent features in hierarchy. 
            lvae = LVAE(in_ch=in_channel,
                    out_ch64=64, out_ch128=128, out_ch256=256, out_ch512=512,
                    kernel1=1, kernel2=2, kernel3=3, padding0=0, padding1=1, stride1=1, stride2=2,
                    flat_dim32=32, flat_dim16=16, flat_dim8=8, flat_dim4=4, flat_dim2=2, flat_dim1=1,
                    latent_dim512=512, latent_dim256=256, latent_dim128=128, latent_dim64=64, latent_dim32=latent_dim,
                    num_class=args.num_classes, dataset=args.dataset, args=args)
            states = torch.load(os.path.join(args.save_path, 'model.pkl'), map_location=args.device)
            lvae.load_state_dict(states['model'])
            lvea=lvae.to(args.device)
            
            test_loader_seen = DataLoader(testdata_seen, batch_size=args.batch_size, shuffle=False, drop_last=False)
            test_loader_unseen = DataLoader(testdata_unseen, batch_size=args.batch_size, shuffle=False, drop_last=False)

            targets, hier_feas = get_hierarchy_feas(test_loader_seen, test_loader_unseen, lvae, args)

            fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
            fig.subpots_adjust(hspace=.5, wspace=.001) 
            axs = axs.ravel()

            for i in range(10): 
                projections = get_tsne(hier_feas[i], 2)
                axs[i].scatter(projections, targets)
                axs[i].set_title("L=%s"%i)
            
            fig.savefig('images/scatter_layers1.png')

        
        if args.reconstruct: 
            lvae = LVAE(in_ch=in_channel,
                    out_ch64=64, out_ch128=128, out_ch256=256, out_ch512=512,
                    kernel1=1, kernel2=2, kernel3=3, padding0=0, padding1=1, stride1=1, stride2=2,
                    flat_dim32=32, flat_dim16=16, flat_dim8=8, flat_dim4=4, flat_dim2=2, flat_dim1=1,
                    latent_dim512=512, latent_dim256=256, latent_dim128=128, latent_dim64=64, latent_dim32=latent_dim,
                    num_class=args.num_classes, dataset=args.dataset, args=args)
            states = torch.load(os.path.join(args.save_path, 'model.pkl'), map_location=args.device)
            lvae.load_state_dict(states['model'])
            lvea=lvae.to(args.device)
            
            test_loader_seen = DataLoader(testdata_seen, batch_size=5, shuffle=False, drop_last=False)
            test_loader_unseen = DataLoader(testdata_unseen, batch_size=5, shuffle=False, drop_last=False)

            show_recon(test_loader_seen, lvae, args)















            
