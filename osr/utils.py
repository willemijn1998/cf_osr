import numpy as np
import os
import shutil
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as T
import pandas as pd


def sample_gaussian(m, v, device):
	sample = torch.randn(m.shape).to(device)
	z = m + (v**0.5)*sample
	return z

def gaussian_parameters(h, dim=-1):
	m, h = torch.split(h, h.size(dim) // 2, dim=dim)
	v = F.softplus(h) + 1e-8
	return m, v

def kl_normal(qm, qv, pm, pv, yh):
	element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm - yh).pow(2) / pv - 1)
	kl = element_wise.sum(-1)
	#print("log var1", qv)
	return kl


def get_exp_name(opt):
	cf_option_str = []

	# if is the baseline
	# if opt.baseline: 
	# 	additional_str = 'baseline'
	# 	cf_option_str.append(additional_str)

	if opt.lamda:
		additional_str = 'lamda%s' %opt.lamda
		cf_option_str.append(additional_str)

	if opt.lr and opt.lr != 0.001:
		additional_str = 'lr%s' %opt.lr
		cf_option_str.append(additional_str)

	if opt.beta_z != 1:
		additional_str = 'betaz%s' %opt.beta_z
		cf_option_str.append(additional_str)

	# if opt.encode_z:
	# 	additional_str = 'encodez%s' %opt.encode_z
	# 	cf_option_str.append(additional_str)
 
	if opt.contrastive_loss:
		additional_str = 'contra%s' %opt.contra_lambda
		cf_option_str.append(additional_str)

	if opt.contrastive_loss and opt.temperature:
		additional_str = 'T%s' %opt.temperature
		cf_option_str.append(additional_str)

	if opt.mmd_loss: 
		additional_str = 'eta%s'%opt.eta
		cf_option_str.append(additional_str)
		if opt.no_aug: 
			additional_str = 'na'
			cf_option_str.append(additional_str)

	if opt.supcon_loss: 
		additional_str = 'theta%s' %opt.theta
		cf_option_str.append(additional_str)
		if opt.no_aug: 
			additional_str = 'na'
			cf_option_str.append(additional_str)

	# if opt.lr_decay:
	# 	additional_str = 'lrdecay%s' %opt.lr_decay
	# 	cf_option_str.append(additional_str)

	if opt.wd != 0.00:
		additional_str = 'wd%s' %opt.wd
		cf_option_str.append(additional_str)

	if opt.rm_skips: 
		additional_str = 'rs'
		cf_option_str.append(additional_str)

	if opt.debug:
		additional_str = 'debug'
		cf_option_str.append(additional_str)
	
	if opt.correct_split: 
		additional_str = 'cs'
		cf_option_str.append(additional_str)

	exp_name = "-".join(cf_option_str)
	exp_name = opt.dataset+'/' + exp_name + '/'+str(opt.exp)

	return exp_name

def get_hparam_table(dict): 
	dfdict = {"Hyperparameters": list(dict.keys()), "Values": list(dict.values())}
	df = pd.DataFrame(dfdict)
	return df.to_markdown()


def mixup_data(x, y, alpha, device):
	"""Returns mixed inputs, pairs of targets, and lambda"""
	if alpha > 0:
		lam = np.random.beta(alpha, alpha)
	else:
		lam = 1

	batch_size = x.size()[0]
	index = torch.randperm(batch_size).to(device)

	mixed_x = lam * x + (1 - lam) * x[index, :]
	y_a, y_b = y, y[index]
	return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
	return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabelSmoothingLoss(nn.Module):
	def __init__(self, classes, smoothing=0.0, dim=-1):
		super(LabelSmoothingLoss, self).__init__()
		self.confidence = 1.0 - smoothing
		self.smoothing = smoothing
		self.cls = classes
		self.dim = dim

	def forward(self, pred, target):
		# pred = pred.log_softmax(dim=self.dim)
		with torch.no_grad():
			# true_dist = pred.data.clone()
			true_dist = torch.zeros_like(pred)
			true_dist.fill_(self.smoothing / (self.cls - 1))
			true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
		return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def MMD(x, y, kernel, device):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

	   Taken from: https://www.kaggle.com/code/onurtunali/maximum-mean-discrepancy/notebook

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
	
    return torch.mean(XX + YY - 2. * XY)

def get_transforms(s, p): 
	transform =  T.Compose([T.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s), 
							T.RandomGrayscale(p=p)])	

	return transform 