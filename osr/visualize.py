from torchvision.utils import make_grid
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T

import torchvision.transforms.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN
from torch.utils.data import Dataset, DataLoader
from torchvision import models, datasets
import numpy as np
import random
import json
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import urllib 
import os
from dataloader import MNIST_Dataset, CIFAR10_Dataset, SVHN_Dataset, CIFARAdd10_Dataset, CIFARAdd50_Dataset, CIFARAddN_Dataset, TinyImageNet_Dataset


plt.rcParams["savefig.bbox"] = 'tight'


# def show(imgs):
#     if not isinstance(imgs, list):
#         imgs = [imgs]
#     fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
#     for i, img in enumerate(imgs):
#         img = img.detach()
#         img = F.to_pil_image(img)
#         axs[0, i].imshow(np.asarray(img))
#         axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

#     plt.show()


# trainset, valset, testset_seen, testset_unseen, channel, seen_classes = get_dataset('CIFAR10', 777, 10, 6, 117)
# loader = data.DataLoader(trainset, batch_size=5, shuffle=False, drop_last=False, num_workers=1, pin_memory=True)
# # transform = T.Compose([T.ColorJitter(0.4, 0.4, 0.4, 0.1), T.RandomGrayscale(p=0.3)])
# transform = T.Compose([T.ColorJitter(0, 0, 0, 0), T.RandomGrayscale(p=0)])


# mean = np.array([0.4914, 0.4822, 0.4465])
# std = np.array([0.2023, 0.1994, 0.2010])

# invTrans = T.Compose([ T.Normalize(mean = [ 0., 0., 0. ],
#                                     std = 1/std),
#                         T.Normalize(mean = -mean,
#                                     std = [ 1., 1., 1. ]),])

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


########### MNIST 
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# trainset = MNIST(root='./data', train=True, download=True, transform=transform)

###### CIFAR10

# mean = np.array([0.4914, 0.4822, 0.4465])
# std = np.array([0.2023, 0.1994, 0.2010])

# invTrans = T.Compose([ T.Normalize(mean = [ 0., 0., 0. ],
#                                     std = 1/std),
#                         T.Normalize(mean = -mean,
#                                     std = [ 1., 1., 1. ]),])
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


# trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
# loader = DataLoader(trainset, batch_size=10, shuffle=True)
# s=0.5 
# p=0.0
# transform = T.Compose([T.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s), 
# 							T.RandomGrayscale(p=p)])

# transform2 = T.Compose([T.RandomGrayscale(p=1)])
# transform = T.Compose([AddGaussianNoise()])
# Options: gaussian blur, random affine, random perspective 

# CIFARAdd10: 
mean = np.array([0.5071, 0.4865, 0.4409])
std = np.array([0.2673, 0.2564, 0.2762])

invTrans = T.Compose([ T.Normalize(mean = [ 0., 0., 0. ],
                                    std = 1/std),
                        T.Normalize(mean = -mean,
                                    std = [ 1., 1., 1. ]),])


dataset = "CIFARAdd10"
seed_sampler = 1234
load_dataset = CIFARAdd10_Dataset(dataset)

train_dataset, val_dataset, test_dataset = load_dataset.sampler(seed_sampler)
testdata_seen, testdata_unseen = test_dataset 
test_loader_unseen = DataLoader(testdata_unseen, batch_size=1, shuffle=False, drop_last=False, num_workers=1, pin_memory=True)
test_loader_seen = DataLoader(testdata_seen, batch_size=1, shuffle=False, drop_last=False, num_workers=1, pin_memory=True)
pred = np.loadtxt('results/CIFARAdd10/lamda100-lr0.0005-eta3.0-cs/1/test_pre_after.txt')
imgs = []

# breakpoint()

for i, (img, lab) in enumerate(test_loader_unseen): 
    if pred[i+2000] == 0: 
        img = img.reshape(3,32,32)
        imgs.append(img)
        plt.imshow(invTrans(img).permute(1,2,0))
        plt.xticks([])
        plt.yticks([])
        plt.savefig('images/class_plane/false{}.png'.format(i))


for i, (img, lab) in enumerate(test_loader_seen): 
    if lab == 0: 
        img = img.reshape(3,32,32)
        imgs.append(img)
        plt.imshow(invTrans(img).permute(1,2,0))
        plt.xticks([])
        plt.yticks([])
        plt.savefig('images/class_plane/true{}.png'.format(i))

# for i, (img, lab) in enumerate(test_loader_unseen): 
#     imgs = torch.cat((invTrans(img), invTrans(transform(img)), invTrans(transform2(img))), 0)
#     grid = make_grid(imgs, nrow=10, ncol=3)
#     plt.imshow(grid.permute(1,2,0))
#     plt.xticks([])
#     plt.yticks([])
#     plt.savefig('images/CIFAR10/plt_imgs1.png', dpi=32)
#     plt.show()
#     break


