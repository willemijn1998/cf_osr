
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.utils import make_grid
from torchvision import transforms as T
from torchvision import models, datasets

from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN

# from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
# from ignite.metrics import Accuracy, Loss, Precision, Recall
# from ignite.handlers import LRScheduler, ModelCheckpoint, global_step_from_engine
# from ignite.contrib.handlers import ProgressBar, TensorboardLogger
# import ignite.contrib.engines.common as common

# import opendatasets as od
import os
from random import randint
# import urllib
# import zipfile

labels_list = []
with open(f'data/tiny-imagenet-200/wnids.txt', 'r') as f:
    for line in f.readlines():
        line = line.strip()
        if len(line) > 0:
            labels_list += [line]

label_word_list = []
label_to_word = {}
with open(f'data/tiny-imagenet-200/words.txt', 'r') as f:
    for line in f.readlines(): 
        line = line.strip('\n')
        label_word = line.split('\t')
        label_word_list += label_word
        label_to_word[label_word[0]] = label_word[1]

words_list = []

for label in labels_list: 
    words_list += [label_to_word[label]]

train_data = datasets.ImageFolder('data/tiny-imagenet-200/train', transform= T.ToTensor())
train_loader = DataLoader(train_data, batch_size=200, shuffle=True)

val_img_dict = {}
with open(f'data/tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
    for line in f.readlines(): 
        line = line.strip('\n')
        label = line.split('\t')
        val_img_dict[label[0]] = label[1]

val_img_dir = 'data/tiny-imagenet-200/val/images'

print(val_img_dict)

for img, folder in val_img_dict.items():
    newpath = (os.path.join(val_img_dir, folder))
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    if os.path.exists(os.path.join(val_img_dir, img)):
        os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))