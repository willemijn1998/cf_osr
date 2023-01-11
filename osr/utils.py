import numpy as np
import os
import shutil
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as T
import pandas as pd
from torch.autograd import Variable


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

	if opt.rec_lamda != 1.0: 
		additional_str = 'rl%s'%opt.rec_lamda
		cf_option_str.append(additional_str)

	if opt.lr and opt.lr != 0.001:
		additional_str = 'lr%s' %opt.lr
		cf_option_str.append(additional_str)
	
	if opt.encode_z != 10: 
		additional_str = 'noZ'
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
		if opt.s_jitter != 0.5: 
			additional_str = 's%s'%opt.s_jitter
			cf_option_str.append(additional_str)

		if opt.p_grayscale != 0.3: 
			additional_str = 'p%s'%opt.p_grayscale
			cf_option_str.append(additional_str)

		if opt.gamma != 1.0: 
			additional_str = 'g%s'%opt.gamma
			cf_option_str.append(additional_str)

		if opt.kernel != "multiscale": 
			additional_str = 'k%s'%opt.kernel
			cf_option_str.append(additional_str)


	if opt.supcon_loss: 
		additional_str = 'theta%s' %opt.theta
		cf_option_str.append(additional_str)
		if opt.no_aug: 
			additional_str = 'na'
			cf_option_str.append(additional_str)
		if opt.s_jitter != 0.5: 
			additional_str = 's%s'%opt.s_jitter
			cf_option_str.append(additional_str)

		if opt.p_grayscale != 0.3: 
			additional_str = 'p%s'%opt.p_grayscale
			cf_option_str.append(additional_str)

		if opt.con_temperature != 0.07: 
			additional_str = 'ct%s'%opt.con_temperature
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
	if opt.dataset == 'CIFARAddN': 
		dataname = 'CIFARAdd{}'.format(str(opt.unseen_num))
	else: 
		dataname = opt.dataset
	exp_name = dataname+'/' + exp_name + '/'+str(opt.exp)

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

class AddGaussianNoise(object):
	def __init__(self, mean=0., std=1., device=torch.device("cuda")):
		self.std = std
		self.mean = mean
		self.device = device 
	
	def __call__(self, tensor):
		return tensor + torch.randn(tensor.size()).to(self.device) * self.std + self.mean
		
	def __repr__(self):
		return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
	

def get_transforms(s, p, dataset): 
	if dataset == "MNIST": 
		transform = T.Compose([AddGaussianNoise()])
	else: 
		transform =  T.Compose([T.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s), 
							T.RandomGrayscale(p=p)])
	return transform 

def get_test_feas(lvae, test_loader_seen, test_loader_unseen, args): 
	for data_test, target_test in test_loader_seen:
		target_test_en = torch.Tensor(target_test.shape[0], args.num_classes)
		target_test_en.zero_()
		target_test_en.scatter_(1, target_test.view(-1, 1), 1)  # one-hot encoding
		target_test_en = target_test_en.to(args.device)
		data_test, target_test = data_test.to(args.device), target_test.to(args.device)
		with torch.no_grad():
			data_test, target_test = Variable(data_test), Variable(target_test)

		mu_test, output_test, de_test = lvae.test(data_test, target_test_en, args)
		output_test = torch.exp(output_test)
		prob_test = output_test.max(1)[0]  # get the value of the max probability
		pre_test = output_test.max(1, keepdim=True)[1]  # get the index of the max log-probability
		rec_test = (de_test - data_test).pow(2).sum((3, 2, 1))
		mu_test = torch.Tensor.cpu(mu_test).detach().numpy()
		target_test = torch.Tensor.cpu(target_test).detach().numpy()
		pre_test = torch.Tensor.cpu(pre_test).detach().numpy()
		rec_test = torch.Tensor.cpu(rec_test).detach().numpy()

		with open('%s/test_fea.txt' % args.save_path, 'ab') as f_test:
			np.savetxt(f_test, mu_test, fmt='%f', delimiter=' ', newline='\r')
			f_test.write(b'\n')
		with open('%s/test_tar.txt' % args.save_path, 'ab') as t_test:
			np.savetxt(t_test, target_test, fmt='%d', delimiter=' ', newline='\r')
			t_test.write(b'\n')
		with open('%s/test_pre.txt' % args.save_path, 'ab') as p_test:
			np.savetxt(p_test, pre_test, fmt='%d', delimiter=' ', newline='\r')
			p_test.write(b'\n')
		with open('%s/test_rec.txt' % args.save_path, 'ab') as l_test:
			np.savetxt(l_test, rec_test, fmt='%f', delimiter=' ', newline='\r')
			l_test.write(b'\n')

		for data_omn, target_omn in test_loader_unseen:
			tar_omn = torch.from_numpy(args.num_classes * np.ones(target_omn.shape[0]))
			data_omn = data_omn.to(args.device) 
			tar_omn = tar_omn.to(args.device)
			with torch.no_grad():
				data_omn = Variable(data_omn)


			mu_omn, output_omn, de_omn = lvae.test(data_omn, target_test_en, args)
			output_omn = torch.exp(output_omn)
			prob_omn = output_omn.max(1)[0]  # get the value of the max probability
			pre_omn = output_omn.max(1, keepdim=True)[1]  # get the index of the max log-probability
			rec_omn = (de_omn - data_omn).pow(2).sum((3, 2, 1))
			mu_omn = torch.Tensor.cpu(mu_omn).detach().numpy()
			tar_omn = torch.Tensor.cpu(tar_omn).detach().numpy()
			pre_omn = torch.Tensor.cpu(pre_omn).detach().numpy()
			rec_omn = torch.Tensor.cpu(rec_omn).detach().numpy()

			with open('%s/test_fea.txt' % args.save_path, 'ab') as f_test:
				np.savetxt(f_test, mu_omn, fmt='%f', delimiter=' ', newline='\r')
				f_test.write(b'\n')
			with open('%s/test_tar.txt' % args.save_path, 'ab') as t_test:
				np.savetxt(t_test, tar_omn, fmt='%d', delimiter=' ', newline='\r')
				t_test.write(b'\n')
			with open('%s/test_pre.txt' % args.save_path, 'ab') as p_test:
				np.savetxt(p_test, pre_omn, fmt='%d', delimiter=' ', newline='\r')
				p_test.write(b'\n')
			with open('%s/test_rec.txt' % args.save_path, 'ab') as l_test:
				np.savetxt(l_test, rec_omn, fmt='%f', delimiter=' ', newline='\r')
				l_test.write(b'\n')

