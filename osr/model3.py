import numpy as np
import torch
import utils as ut
from torch.autograd import Variable
from torch import autograd, nn, optim
from torch.nn import functional as F

# Model that classifies according to loglikelihood 

reconstruction_function = nn.MSELoss()
reconstruction_function.size_average = False
nllloss = nn.NLLLoss()

class CONV(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, padding, stride, flat_dim, latent_dim):
        super(CONV, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.flat_dim = flat_dim
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding, stride=stride),  # (w-k+2p)/s+1
            nn.BatchNorm2d(out_ch, affine=False),
            nn.PReLU()
        )
        self.mean_layer = nn.Sequential(
            nn.Linear(out_ch*flat_dim*flat_dim, latent_dim)
        )
        self.var_layer = nn.Sequential(
            nn.Linear(out_ch*flat_dim*flat_dim, latent_dim)
        )

    def encode(self, x):
        h = self.net(x)
        h_flat = h.view(-1, self.out_ch*self.flat_dim*self.flat_dim)
        mu, var = self.mean_layer(h_flat), self.var_layer(h_flat)
        var = F.softplus(var) + 1e-8
        # mu, var = ut.gaussian_parameters(h, dim=1)
        return h, mu, var

class TCONV(nn.Module):
    def __init__(self, in_size, unflat_dim, t_in_ch, t_out_ch, t_kernel, t_padding, t_stride, out_dim, t_latent_dim):
        super(TCONV, self).__init__()
        self.in_size = in_size
        self.unflat_dim = unflat_dim
        self.t_in_ch = t_in_ch
        self.t_out_ch = t_out_ch
        self.t_kernel = t_kernel
        self.t_stride = t_stride
        self.t_padding = t_padding
        self.out_dim = out_dim
        self.t_latent_dim = t_latent_dim

        self.fc = nn.Linear(in_size, t_in_ch * unflat_dim * unflat_dim)
        self.net = nn.Sequential(
            nn.PReLU(),
            nn.ConvTranspose2d(t_in_ch, t_out_ch, kernel_size=t_kernel, padding=t_padding, stride=t_stride),  # (w-k+2p)/s+1
            nn.BatchNorm2d(t_out_ch, affine=False),
        )
        self.mean_layer = nn.Sequential(
            nn.Linear(t_out_ch*out_dim*out_dim, t_latent_dim)
        )
        self.var_layer = nn.Sequential(
            nn.Linear(t_out_ch*out_dim*out_dim, t_latent_dim)
        )

    def decode(self, x):
        x = self.fc(x)
        x = x.view(-1, self.t_in_ch, self.unflat_dim, self.unflat_dim) # 5_1: 512, 1, 1
        h = self.net(x) # 5_1: 512, 2, 2 
        h_flat = h.view(-1, self.t_out_ch * self.out_dim * self.out_dim)
        mu, var = self.mean_layer(h_flat), self.var_layer(h_flat)
        var = F.softplus(var) + 1e-8
        # mu, var = ut.gaussian_parameters(h, dim=1)
        return h, mu, var

class FCONV(nn.Module):
    def __init__(self, in_size, unflat_dim, t_in_ch, t_out_ch, t_kernel, t_padding, t_stride):
        super(FCONV, self).__init__()
        self.in_size = in_size
        self.unflat_dim = unflat_dim
        self.t_in_ch = t_in_ch
        self.t_out_ch = t_out_ch
        self.t_kernel = t_kernel
        self.t_stride = t_stride
        self.t_padding = t_padding

        self.fc_final = nn.Linear(in_size, t_in_ch * unflat_dim * unflat_dim)
        self.final = nn.Sequential(
            nn.PReLU(),
            nn.ConvTranspose2d(t_in_ch, t_out_ch, kernel_size=t_kernel, padding=t_padding, stride=t_stride),  # (w-k+2p)/s+1
            #nn.Sigmoid()
            nn.Tanh()
        )

    def final_decode(self,x):
        x = self.fc_final(x)
        x = x.view(-1, self.t_in_ch, self.unflat_dim, self.unflat_dim)
        x_re = self.final(x)
        return x_re

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.2):
        # base_temperature used to be 0.07
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...]. n_views is number of augmentations
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)+ 1e-6)
        # Changed to log instead of exp because not stable 
        # log_logits = logits * logits_mask
        # log_prob = logits - log_logits.sum(1, keepdim=True)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

class LVAE(nn.Module):
    def __init__(self, in_ch=3,
                 out_ch64=64, out_ch128=128, out_ch256=256, out_ch512=512,
                 kernel1=1, kernel2=2, kernel3=3, padding0=0, padding1=1, padding2=2, stride1=1, stride2=2,
                 flat_dim32=32, flat_dim16=16, flat_dim8=8, flat_dim4=4, flat_dim2=2, flat_dim1=1,
                 latent_dim512=512, latent_dim256=256, latent_dim128=128, latent_dim64=64, latent_dim32=32, num_class =15,
                 dataset="MNIST", args=None):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch64 = out_ch64
        self.out_ch128 = out_ch128
        self.out_ch256 = out_ch256
        self.out_ch512 = out_ch512
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.kernel3 = kernel3
        self.padding0 = padding0
        self.padding1 = padding1
        self.padding2 = padding2
        self.stride1 = stride1
        self.stride2 = stride2
        self.flat_dim32 = flat_dim32
        self.flat_dim16 = flat_dim16
        self.flat_dim8 = flat_dim8
        self.flat_dim4 = flat_dim4
        self.flat_dim2 = flat_dim2
        self.flat_dim1 = flat_dim1
        self.latent_dim512 = latent_dim512
        self.latent_dim256 = latent_dim256
        self.latent_dim128 = latent_dim128
        self.latent_dim64 = latent_dim64
        self.latent_dim32 = latent_dim32 # 42
        self.num_class = num_class
        self.dataset = dataset

        # initialize required CONVs
        if dataset == "MNIST":
            self.CONV1_1 = CONV(self.in_ch, self.out_ch64, self.kernel1, self.padding2, self.stride1, self.flat_dim32,
                                self.latent_dim512)
        else:
            self.CONV1_1 = CONV(self.in_ch, self.out_ch64, self.kernel1, self.padding0, self.stride1, self.flat_dim32,
                                self.latent_dim512)
        self.CONV1_2 = CONV(self.out_ch64, self.out_ch64, self.kernel3, self.padding1, self.stride2, self.flat_dim16,
                            self.latent_dim512)

        self.CONV2_1 = CONV(self.out_ch64, self.out_ch128, self.kernel3, self.padding1, self.stride1, self.flat_dim16, self.latent_dim256)
        self.CONV2_2 = CONV(self.out_ch128, self.out_ch128, self.kernel3, self.padding1, self.stride2, self.flat_dim8, self.latent_dim256)

        self.CONV3_1 = CONV(self.out_ch128, self.out_ch256, self.kernel3, self.padding1, self.stride1, self.flat_dim8,
                            self.latent_dim128)
        self.CONV3_2 = CONV(self.out_ch256, self.out_ch256, self.kernel3, self.padding1, self.stride2, self.flat_dim4,
                            self.latent_dim128)

        self.CONV4_1 = CONV(self.out_ch256, self.out_ch512, self.kernel3, self.padding1, self.stride1, self.flat_dim4,
                            self.latent_dim64)
        self.CONV4_2 = CONV(self.out_ch512, self.out_ch512, self.kernel3, self.padding1, self.stride2, self.flat_dim2,
                            self.latent_dim64)

        self.CONV5_1 = CONV(self.out_ch512, self.out_ch512, self.kernel3, self.padding1, self.stride1, self.flat_dim2,
                            self.latent_dim32)
        self.CONV5_2 = CONV(self.out_ch512, self.out_ch512, self.kernel3, self.padding1, self.stride2, self.flat_dim1,
                            self.latent_dim32)

        # initialize required TCONVs
        self.TCONV5_2 = TCONV(self.latent_dim32, self.flat_dim1, self.out_ch512, self.out_ch512, self.kernel2,
                              self.padding0, self.stride2, self.flat_dim2, self.latent_dim32)
        self.TCONV5_1 = TCONV(self.latent_dim32, self.flat_dim2, self.out_ch512, self.out_ch512, self.kernel1,
                              self.padding0, self.stride1, self.flat_dim2, self.latent_dim64)

        self.TCONV4_2 = TCONV(self.latent_dim64, self.flat_dim2, self.out_ch512, self.out_ch512, self.kernel2,
                              self.padding0, self.stride2, self.flat_dim4, self.latent_dim64)
        self.TCONV4_1 = TCONV(self.latent_dim64, self.flat_dim4, self.out_ch512, self.out_ch256, self.kernel1,
                              self.padding0, self.stride1, self.flat_dim4, self.latent_dim128)

        self.TCONV3_2 = TCONV(self.latent_dim128, self.flat_dim4, self.out_ch256, self.out_ch256, self.kernel2,
                              self.padding0, self.stride2, self.flat_dim8, self.latent_dim128)
        self.TCONV3_1 = TCONV(self.latent_dim128, self.flat_dim8, self.out_ch256, self.out_ch128, self.kernel1,
                              self.padding0, self.stride1, self.flat_dim8, self.latent_dim256)

        self.TCONV2_2 = TCONV(self.latent_dim256, self.flat_dim8, self.out_ch128, self.out_ch128, self.kernel2,
                              self.padding0, self.stride2, self.flat_dim16, self.latent_dim256)
        self.TCONV2_1 = TCONV(self.latent_dim256, self.flat_dim16, self.out_ch128, self.out_ch64, self.kernel1,
                              self.padding0, self.stride1, self.flat_dim16, self.latent_dim512)

        self.TCONV1_2 = TCONV(self.latent_dim512, self.flat_dim16, self.out_ch64, self.out_ch64, self.kernel2,
                              self.padding0, self.stride2, self.flat_dim32, self.latent_dim512)
        if dataset == "MNIST":
            self.TCONV1_1 = FCONV(self.latent_dim512, self.flat_dim32, self.out_ch64, self.in_ch, self.kernel1,
                                self.padding2, self.stride1)
        else:
            self.TCONV1_1 = FCONV(self.latent_dim512, self.flat_dim32, self.out_ch64, self.in_ch, self.kernel1,
                                self.padding0, self.stride1)

        ## ugly add by WT
        self.classifier = nn.Linear(32, self.num_class)
        self.one_hot = nn.Linear(self.num_class, 32)


    def lnet(self, x, y_de, args):
        # ---deterministic upward pass
        # upwards
        enc1_1, mu_up1_1, var_up1_1 = self.CONV1_1.encode(x)
        enc1_2, mu_up1_2, var_up1_2 = self.CONV1_2.encode(enc1_1)

        enc2_1, mu_up2_1, var_up2_1 = self.CONV2_1.encode(enc1_2)
        enc2_2, mu_up2_2, var_up2_2 = self.CONV2_2.encode(enc2_1)

        enc3_1, mu_up3_1, var_up3_1 = self.CONV3_1.encode(enc2_2)
        enc3_2, mu_up3_2, var_up3_2 = self.CONV3_2.encode(enc3_1)

        enc4_1, mu_up4_1, var_up4_1 = self.CONV4_1.encode(enc3_2)
        enc4_2, mu_up4_2, var_up4_2 = self.CONV4_2.encode(enc4_1)

        enc5_1, mu_up5_1, var_up5_1 = self.CONV5_1.encode(enc4_2)
        enc5_2, mu_latent, var_latent = self.CONV5_2.encode(enc5_1)

        # enc5_2.shape = (64, 512, 1, 1)
        # mu, var.shape =(64, 42)

        # split z and y
        if args.encode_z:
            z_latent_mu, y_latent_mu = mu_latent.split([args.encode_z, 32], dim=1)
            z_latent_var, y_latent_var = var_latent.split([args.encode_z, 32], dim=1)
            latent = ut.sample_gaussian(mu_latent, var_latent, args.device)
            latent_y = ut.sample_gaussian(y_latent_mu, y_latent_var, args.device)
        else:
            y_latent_mu = mu_latent
            y_latent_var = var_latent
            latent = ut.sample_gaussian(mu_latent, var_latent, args.device)
            latent_y = latent


        predict = F.log_softmax(self.classifier(latent_y), dim=1)
        predict_test = F.log_softmax(self.classifier(y_latent_mu), dim=1)
        yh = self.one_hot(y_de)

        # partially downwards
        dec5_1, mu_dn5_1, var_dn5_1 = self.TCONV5_2.decode(latent) # 64, 42
        # (512, 2, 2)
        prec_up5_1 = var_up5_1 ** (-1)
        prec_dn5_1 = var_dn5_1 ** (-1)
        qmu5_1 = (mu_up5_1 * prec_up5_1 + mu_dn5_1 * prec_dn5_1) / (prec_up5_1 + prec_dn5_1)
        qvar5_1 = (prec_up5_1 + prec_dn5_1) ** (-1)
        de_latent5_1 = ut.sample_gaussian(qmu5_1, qvar5_1, args.device)

        dec4_2, mu_dn4_2, var_dn4_2 = self.TCONV5_1.decode(de_latent5_1) # 
        # 512, 2, 2
        prec_up4_2 = var_up4_2 ** (-1)
        prec_dn4_2 = var_dn4_2 ** (-1)
        qmu4_2 = (mu_up4_2 * prec_up4_2 + mu_dn4_2 * prec_dn4_2) / (prec_up4_2 + prec_dn4_2)
        qvar4_2 = (prec_up4_2 + prec_dn4_2) ** (-1)
        de_latent4_2 = ut.sample_gaussian(qmu4_2, qvar4_2, args.device)

        dec4_1, mu_dn4_1, var_dn4_1 = self.TCONV4_2.decode(de_latent4_2)
        # 512, 4, 4 
        prec_up4_1 = var_up4_1 ** (-1)
        prec_dn4_1 = var_dn4_1 ** (-1)
        qmu4_1 = (mu_up4_1 * prec_up4_1 + mu_dn4_1 * prec_dn4_1) / (prec_up4_1 + prec_dn4_1)
        qvar4_1 = (prec_up4_1 + prec_dn4_1) ** (-1)
        de_latent4_1 = ut.sample_gaussian(qmu4_1, qvar4_1, args.device)

        dec3_2, mu_dn3_2, var_dn3_2 = self.TCONV4_1.decode(de_latent4_1)
        # 256, 4, 4
        prec_up3_2 = var_up3_2 ** (-1)
        prec_dn3_2 = var_dn3_2 ** (-1)
        qmu3_2 = (mu_up3_2 * prec_up3_2 + mu_dn3_2 * prec_dn3_2) / (prec_up3_2 + prec_dn3_2)
        qvar3_2 = (prec_up3_2 + prec_dn3_2) ** (-1)
        de_latent3_2 = ut.sample_gaussian(qmu3_2, qvar3_2, args.device)

        dec3_1, mu_dn3_1, var_dn3_1 = self.TCONV3_2.decode(de_latent3_2)
        # 256, 8, 8
        prec_up3_1 = var_up3_1 ** (-1)
        prec_dn3_1 = var_dn3_1 ** (-1)
        qmu3_1 = (mu_up3_1 * prec_up3_1 + mu_dn3_1 * prec_dn3_1) / (prec_up3_1 + prec_dn3_1)
        qvar3_1 = (prec_up3_1 + prec_dn3_1) ** (-1)
        de_latent3_1 = ut.sample_gaussian(qmu3_1, qvar3_1, args.device)

        dec2_2, mu_dn2_2, var_dn2_2 = self.TCONV3_1.decode(de_latent3_1)
        # 128, 8, 8
        prec_up2_2 = var_up2_2 ** (-1)
        prec_dn2_2 = var_dn2_2 ** (-1)
        qmu2_2 = (mu_up2_2 * prec_up2_2 + mu_dn2_2 * prec_dn2_2) / (prec_up2_2 + prec_dn2_2)
        qvar2_2 = (prec_up2_2 + prec_dn2_2) ** (-1)
        de_latent2_2 = ut.sample_gaussian(qmu2_2, qvar2_2, args.device)

        dec2_1, mu_dn2_1, var_dn2_1 = self.TCONV2_2.decode(de_latent2_2)
        # 128, 16, 16 
        prec_up2_1 = var_up2_1 ** (-1)
        prec_dn2_1 = var_dn2_1 ** (-1)
        qmu2_1 = (mu_up2_1 * prec_up2_1 + mu_dn2_1 * prec_dn2_1) / (prec_up2_1 + prec_dn2_1)
        qvar2_1 = (prec_up2_1 + prec_dn2_1) ** (-1)
        de_latent2_1 = ut.sample_gaussian(qmu2_1, qvar2_1, args.device)

        dec1_2, mu_dn1_2, var_dn1_2 = self.TCONV2_1.decode(de_latent2_1)
        # 64, 16, 16 
        prec_up1_2 = var_up1_2 ** (-1)
        prec_dn1_2 = var_dn1_2 ** (-1)
        qmu1_2 = (mu_up1_2 * prec_up1_2 + mu_dn1_2 * prec_dn1_2) / (prec_up1_2 + prec_dn1_2)
        qvar1_2 = (prec_up1_2 + prec_dn1_2) ** (-1)
        de_latent1_2 = ut.sample_gaussian(qmu1_2, qvar1_2, args.device)

        dec1_1, mu_dn1_1, var_dn1_1 = self.TCONV1_2.decode(de_latent1_2)
        # 64, 32, 32
        prec_up1_1 = var_up1_1 ** (-1)
        prec_dn1_1 = var_dn1_1 ** (-1)
        qmu1_1 = (mu_up1_1 * prec_up1_1 + mu_dn1_1 * prec_dn1_1) / (prec_up1_1 + prec_dn1_1)
        qvar1_1 = (prec_up1_1 + prec_dn1_1) ** (-1)
        de_latent1_1 = ut.sample_gaussian(qmu1_1, qvar1_1, args.device)

        x_re = self.TCONV1_1.final_decode(de_latent1_1)
        # 3, 32, 32 

        if args.contrastive_loss and self.training and not args.eval:
            self.contra_loss = self.contrastive_loss(x, y_de, x_re, args)

        return latent, mu_latent, var_latent, \
               qmu5_1, qvar5_1, qmu4_2, qvar4_2, qmu4_1, qvar4_1, qmu3_2, qvar3_2, qmu3_1, qvar3_1, \
               qmu2_2, qvar2_2, qmu2_1, qvar2_1, qmu1_2, qvar1_2, qmu1_1, qvar1_1, \
               predict, predict_test, yh, \
               x_re, \
               mu_dn5_1, var_dn5_1, mu_dn4_2, var_dn4_2, mu_dn4_1, var_dn4_1, mu_dn3_2, var_dn3_2, mu_dn3_1, var_dn3_1, \
               mu_dn2_2, var_dn2_2, mu_dn2_1, var_dn2_1, mu_dn1_2, var_dn1_2, mu_dn1_1, var_dn1_1

    def loss(self, x, y, y_de, beta, lamda, args):

        latent, mu_latent, var_latent, \
        qmu5_1, qvar5_1, qmu4_2, qvar4_2, qmu4_1, qvar4_1, qmu3_2, qvar3_2, qmu3_1, qvar3_1, \
        qmu2_2, qvar2_2, qmu2_1, qvar2_1, qmu1_2, qvar1_2, qmu1_1, qvar1_1, \
        predict, predict_test, yh, \
        x_re, \
        pmu5_1, pvar5_1,pmu4_2, pvar4_2, pmu4_1, pvar4_1, pmu3_2, pvar3_2, pmu3_1, pvar3_1, \
        pmu2_2, pvar2_2, pmu2_1, pvar2_1, pmu1_2, pvar1_2, pmu1_1, pvar1_1 = self.lnet(x, y_de, args)

        rec = reconstruction_function(x_re, x)

        # split z and y if encode_z
        if args.encode_z:
            z_latent_mu, y_latent_mu = mu_latent.split([args.encode_z, 32], dim=1)
            z_latent_var, y_latent_var = var_latent.split([args.encode_z, 32], dim=1)
            pm_z, pv_z = torch.zeros(z_latent_mu.shape).to(args.device), torch.ones(z_latent_var.shape).to(args.device)
        else:
            y_latent_mu = mu_latent
            y_latent_var = var_latent

        pm, pv = torch.zeros(y_latent_mu.shape).to(args.device), torch.ones(y_latent_var.shape).to(args.device)
        # print("mu1", mu1)
        kl_latent = ut.kl_normal(y_latent_mu, y_latent_var, pm, pv, yh)
        kl5_1 = ut.kl_normal(qmu5_1, qvar5_1, pmu5_1, pvar5_1, 0)
        kl4_2 = ut.kl_normal(qmu4_2, qvar4_2, pmu4_2, pvar4_2, 0)
        kl4_1 = ut.kl_normal(qmu4_1, qvar4_1, pmu4_1, pvar4_1, 0)
        kl3_2 = ut.kl_normal(qmu3_2, qvar3_2, pmu3_2, pvar3_2, 0)
        kl3_1 = ut.kl_normal(qmu3_1, qvar3_1, pmu3_1, pvar3_1, 0)
        kl2_2 = ut.kl_normal(qmu2_2, qvar2_2, pmu2_2, pvar2_2, 0)  
        kl2_1 = ut.kl_normal(qmu2_1, qvar2_1, pmu2_1, pvar2_1, 0)
        kl1_2 = ut.kl_normal(qmu1_2, qvar1_2, pmu1_2, pvar1_2, 0)
        kl1_1 = ut.kl_normal(qmu1_1, qvar1_1, pmu1_1, pvar1_1, 0)

        kl_all = kl_latent + kl5_1 + kl4_2 + kl4_1 + kl3_2 + kl3_1 + kl2_2 + kl2_1 + kl1_2 + kl1_1

        if args.encode_z:
            kl_all += args.beta_z * ut.kl_normal(z_latent_mu, z_latent_var, pm_z, pv_z, 0)

        kl = beta * torch.mean(kl_all)

        ce = nllloss(predict, y)

        nelbo = args.rec_lamda * rec + kl + lamda*ce

        if args.contrastive_loss:
            contra_loss = self.contra_loss
            nelbo += contra_loss

        if args.mmd_loss: 
            if args.no_aug: 
                x_aug = x
            else: 
                x_aug = args.transforms(x)
            self.invar_loss = self.mmd_loss(y_latent_mu, x_aug, y, y_de, args)
            nelbo += args.eta * self.invar_loss

        if args.supcon_loss: 
            if args.no_aug: 
                x_aug = x
            else: 
                x_aug = args.transforms(x)

            W_aug, _, _= self.test(x_aug, y_de, args)
            WW_aug = torch.cat((y_latent_mu.unsqueeze(dim=1), W_aug.unsqueeze(dim=1)), dim=1)
            self.supcon_loss = self.supcon_critic(WW_aug, y)
            nelbo += args.theta * self.supcon_loss 

        # nelbo = rec
        return nelbo, y_latent_mu, predict, predict_test, x_re,rec,kl,lamda*ce


    def test(self, x, y_de, args):
        _, mu_latent, _, \
        _, _, _, _, _, _, _, _, _, _,\
        _, _, _, _, _, _, _, _, \
        _, predict_test, _ ,\
        x_re, \
        pmu5_1, pvar5_1, pmu4_2, pvar4_2, pmu4_1, pvar4_1, pmu3_2, pvar3_2, pmu3_1, pvar3_1, \
        pmu2_2, pvar2_2, pmu2_1, pvar2_1, pmu1_2, pvar1_2, pmu1_1, pvar1_1 = self.lnet(x, y_de, args)

        if args.encode_z:
            z_latent_mu, y_latent_mu = mu_latent.split([args.encode_z, 32], dim=1)
        else:
            y_latent_mu = mu_latent

        return y_latent_mu, predict_test, x_re


    def test(self, x, y_de, args):
        _, mu_latent, _, \
        _, _, _, _, _, _, _, _, _, _,\
        _, _, _, _, _, _, _, _, \
        _, predict_test, _ ,\
        x_re, \
        pmu5_1, pvar5_1, pmu4_2, pvar4_2, pmu4_1, pvar4_1, pmu3_2, pvar3_2, pmu3_1, pvar3_1, \
        pmu2_2, pvar2_2, pmu2_1, pvar2_1, pmu1_2, pvar1_2, pmu1_1, pvar1_1 = self.lnet(x, y_de, args)

        if args.encode_z:
            z_latent_mu, y_latent_mu = mu_latent.split([args.encode_z, 32], dim=1)
        else:
            y_latent_mu = mu_latent

        return y_latent_mu, predict_test, x_re

    def get_yh(self, y_de):
        yh = self.one_hot(y_de)
        return yh

    def contrastive_loss(self, x, target, rec_x, args):
        """
        z : batchsize * 10
        """
        bs = x.size(0)
        ### get current yh for each class
        target_en = torch.eye(args.num_classes)
        class_yh = self.get_yh(target_en.to(args.device)) # 6*32
        yh_size = class_yh.size(1)

        neg_class_num = args.num_classes - 1
        # z_neg = z.unsqueeze(1).repeat(1, neg_class_num, 1)
        y_neg = torch.zeros((bs, neg_class_num, yh_size)).to(args.device)
        for i in range(bs):
            y_sample = [idx for idx in range(args.num_classes) if idx != torch.argmax(target[i])]
            y_neg[i] = class_yh[y_sample]
        # zy_neg = torch.cat([z_neg, y_neg], dim=2).view(bs*neg_class_num, z.size(1)+yh_size)

        rec_x_neg = self.generate_cf(x, target, y_neg, args)
        rec_x_all = torch.cat([rec_x.unsqueeze(1), rec_x_neg], dim=1)

        x_expand = x.unsqueeze(1).repeat(1, args.num_classes, 1, 1, 1)
        neg_dist = -((x_expand - rec_x_all) ** 2).mean((2,3,4)) * args.temperature  # N*(K+1)
        label = torch.zeros(bs).to(args.device).long()
        contrastive_loss_euclidean = nn.CrossEntropyLoss()(neg_dist, label)

        return contrastive_loss_euclidean

    def mmd_loss(self, W, x_aug, labels, labels_onehot, args): 
        "MMD loss between input and augmented input"    
        W_aug, _, _ = self.test(x_aug, labels_onehot, args)

        mmd_losses = torch.zeros(args.num_classes)

        for i in range(args.num_classes): 
            if len(W[labels==i]) == 0 or len(W_aug[labels==i]) == 0: 
                continue
            else: 
                N = len(labels[labels==i])
            mmd_losses[i] = N * ut.MMD(W[labels==i], W_aug[labels==i], kernel=args.kernel, device=args.device)
        
        return torch.mean(mmd_losses)


    def rec_loss_cf(self, feature_y_mean, test_loader_seen, test_loader_unseen, args):
        rec_loss_cf_all = []
        class_num = feature_y_mean.size(0)
        for data_test, target_test in test_loader_seen:
            target_test_en = torch.Tensor(target_test.shape[0], args.num_classes)
            target_test_en.zero_()
            target_test_en.scatter_(1, target_test.view(-1, 1), 1)  # one-hot encoding
            target_test_en = target_test_en.to(args.device)
            data_test, target_test = data_test.to(args.device), target_test.to(args.device)
            with torch.no_grad():
                data_test, target_test = Variable(data_test), Variable(target_test)

            re_test = self.generate_cf(data_test, target_test_en, feature_y_mean, args)
            data_test_cf = data_test.unsqueeze(1).repeat(1, class_num, 1, 1, 1)
            rec_loss = (re_test - data_test_cf).pow(2).sum((2, 3, 4))
            rec_loss_cf = rec_loss.min(1)[0]
            rec_loss_cf_all.append(rec_loss_cf)


        for data_test, target_test in test_loader_unseen:
            target_test_en = torch.Tensor(target_test.shape[0], args.num_classes)
            target_test_en.zero_()
            # target_test_en.scatter_(1, target_test.view(-1, 1), 1)  # one-hot encoding
            target_test_en = target_test_en.to(args.device)
            data_test, target_test = data_test.to(args.device), target_test.to(args.device)
            with torch.no_grad():
                data_test, target_test = Variable(data_test), Variable(target_test)

            re_test = self.generate_cf(data_test, target_test_en, feature_y_mean, args)
            data_test_cf = data_test.unsqueeze(1).repeat(1, class_num, 1, 1, 1)
            rec_loss = (re_test - data_test_cf).pow(2).sum((2, 3, 4))
            rec_loss_cf = rec_loss.min(1)[0]
            rec_loss_cf_all.append(rec_loss_cf)

        rec_loss_cf_all = torch.cat(rec_loss_cf_all, 0)
        return rec_loss_cf_all


    def rec_loss_cf_train(self, feature_y_mean, train_loader, args):
        rec_loss_cf_all = []
        class_num = feature_y_mean.size(0)
        for data_train, target_train in train_loader:
            target_train_en = torch.Tensor(target_train.shape[0], args.num_classes)
            target_train_en.zero_()
            target_train_en.scatter_(1, target_train.view(-1, 1), 1)  # one-hot encoding
            target_train_en = target_train_en.to(args.device)
            data_train, target_train = data_train.to(args.device), target_train.to(args.device)
            with torch.no_grad():
                data_train, target_train = Variable(data_train), Variable(target_train)

            re_train = self.generate_cf(data_train, target_train_en, feature_y_mean, args)
            data_train_cf = data_train.unsqueeze(1).repeat(1, class_num, 1, 1, 1)
            rec_loss = (re_train - data_train_cf).pow(2).sum((2, 3, 4))
            rec_loss_cf = rec_loss.min(1)[0]
            rec_loss_cf_all.append(rec_loss_cf)

        rec_loss_cf_all = torch.cat(rec_loss_cf_all, 0)
        return rec_loss_cf_all

    def generate_sample_feas(self, x, args): 
        enc1_1, mu_up1_1, var_up1_1 = self.CONV1_1.encode(x)
        enc1_2, mu_up1_2, var_up1_2 = self.CONV1_2.encode(enc1_1)

        enc2_1, mu_up2_1, var_up2_1 = self.CONV2_1.encode(enc1_2)
        enc2_2, mu_up2_2, var_up2_2 = self.CONV2_2.encode(enc2_1)

        enc3_1, mu_up3_1, var_up3_1 = self.CONV3_1.encode(enc2_2)
        enc3_2, mu_up3_2, var_up3_2 = self.CONV3_2.encode(enc3_1)

        enc4_1, mu_up4_1, var_up4_1 = self.CONV4_1.encode(enc3_2)
        enc4_2, mu_up4_2, var_up4_2 = self.CONV4_2.encode(enc4_1)

        enc5_1, mu_up5_1, var_up5_1 = self.CONV5_1.encode(enc4_2)
        enc5_2, mu_latent, var_latent = self.CONV5_2.encode(enc5_1)


        z_latent_mu, _ = mu_latent.split([args.encode_z, 32], dim=1)
        z_latent_var, _ = var_latent.split([args.encode_z, 32], dim=1)

        return z_latent_mu, z_latent_var


    def generate_cf(self, x, y_de, mean_y, args):
        """
        :param x:
        :param mean_y: list, the class-wise feature y
        This function 
        """
        if mean_y.dim() == 2:
            class_num = mean_y.size(0)
        elif mean_y.dim() == 3:
            class_num = mean_y.size(1)
        bs = x.size(0)

        enc1_1, mu_up1_1, var_up1_1 = self.CONV1_1.encode(x)
        enc1_2, mu_up1_2, var_up1_2 = self.CONV1_2.encode(enc1_1)

        enc2_1, mu_up2_1, var_up2_1 = self.CONV2_1.encode(enc1_2)
        enc2_2, mu_up2_2, var_up2_2 = self.CONV2_2.encode(enc2_1)

        enc3_1, mu_up3_1, var_up3_1 = self.CONV3_1.encode(enc2_2)
        enc3_2, mu_up3_2, var_up3_2 = self.CONV3_2.encode(enc3_1)

        enc4_1, mu_up4_1, var_up4_1 = self.CONV4_1.encode(enc3_2)
        enc4_2, mu_up4_2, var_up4_2 = self.CONV4_2.encode(enc4_1)

        enc5_1, mu_up5_1, var_up5_1 = self.CONV5_1.encode(enc4_2)
        enc5_2, mu_latent, var_latent = self.CONV5_2.encode(enc5_1)


        z_latent_mu, y_latent_mu = mu_latent.split([args.encode_z, 32], dim=1)
        z_latent_var, y_latent_var = var_latent.split([args.encode_z, 32], dim=1)

        z_latent_mu = z_latent_mu.unsqueeze(1).repeat(1, class_num, 1)
        if mean_y.dim() == 2:
            y_mu =mean_y.unsqueeze(0).repeat(bs, 1, 1)
        elif mean_y.dim() == 3:
            y_mu = mean_y
        latent_zy = torch.cat([z_latent_mu, y_mu], dim=2).view(bs*class_num, mu_latent.size(1))

        # latent = ut.sample_gaussian(mu_latent, var_latent)

        # partially downwards
        dec5_1, mu_dn5_1, var_dn5_1 = self.TCONV5_2.decode(latent_zy)
        prec_up5_1 = (var_up5_1 ** (-1)).repeat(class_num, 1)
        prec_dn5_1 = var_dn5_1 ** (-1)
        qmu5_1 = (mu_up5_1.repeat(class_num, 1) * prec_up5_1 + mu_dn5_1 * prec_dn5_1) / (prec_up5_1 + prec_dn5_1)
        qvar5_1 = (prec_up5_1 + prec_dn5_1) ** (-1)
        de_latent5_1 = ut.sample_gaussian(qmu5_1, qvar5_1, args.device)

        dec4_2, mu_dn4_2, var_dn4_2 = self.TCONV5_1.decode(de_latent5_1)
        prec_up4_2 = (var_up4_2 ** (-1)).repeat(class_num, 1)
        prec_dn4_2 = var_dn4_2 ** (-1)
        qmu4_2 = (mu_up4_2.repeat(class_num, 1) * prec_up4_2 + mu_dn4_2 * prec_dn4_2) / (prec_up4_2 + prec_dn4_2)
        qvar4_2 = (prec_up4_2 + prec_dn4_2) ** (-1)
        de_latent4_2 = ut.sample_gaussian(qmu4_2, qvar4_2, args.device)

        dec4_1, mu_dn4_1, var_dn4_1 = self.TCONV4_2.decode(de_latent4_2)
        prec_up4_1 = (var_up4_1 ** (-1)).repeat(class_num, 1)
        prec_dn4_1 = var_dn4_1 ** (-1)
        qmu4_1 = (mu_up4_1.repeat(class_num, 1) * prec_up4_1 + mu_dn4_1 * prec_dn4_1) / (prec_up4_1 + prec_dn4_1)
        qvar4_1 = (prec_up4_1 + prec_dn4_1) ** (-1)
        de_latent4_1 = ut.sample_gaussian(qmu4_1, qvar4_1, args.device)

        dec3_2, mu_dn3_2, var_dn3_2 = self.TCONV4_1.decode(de_latent4_1)
        prec_up3_2 = (var_up3_2 ** (-1)).repeat(class_num, 1)
        prec_dn3_2 = var_dn3_2 ** (-1)
        qmu3_2 = (mu_up3_2.repeat(class_num, 1) * prec_up3_2 + mu_dn3_2 * prec_dn3_2) / (prec_up3_2 + prec_dn3_2)
        qvar3_2 = (prec_up3_2 + prec_dn3_2) ** (-1)
        de_latent3_2 = ut.sample_gaussian(qmu3_2, qvar3_2, args.device)

        dec3_1, mu_dn3_1, var_dn3_1 = self.TCONV3_2.decode(de_latent3_2)
        prec_up3_1 = (var_up3_1 ** (-1)).repeat(class_num, 1)
        prec_dn3_1 = var_dn3_1 ** (-1)
        qmu3_1 = (mu_up3_1.repeat(class_num, 1) * prec_up3_1 + mu_dn3_1 * prec_dn3_1) / (prec_up3_1 + prec_dn3_1)
        qvar3_1 = (prec_up3_1 + prec_dn3_1) ** (-1)
        de_latent3_1 = ut.sample_gaussian(qmu3_1, qvar3_1, args.device)

        dec2_2, mu_dn2_2, var_dn2_2 = self.TCONV3_1.decode(de_latent3_1)
        prec_up2_2 = (var_up2_2 ** (-1)).repeat(class_num, 1)
        prec_dn2_2 = var_dn2_2 ** (-1)
        qmu2_2 = (mu_up2_2.repeat(class_num, 1) * prec_up2_2 + mu_dn2_2 * prec_dn2_2) / (prec_up2_2 + prec_dn2_2)
        qvar2_2 = (prec_up2_2 + prec_dn2_2) ** (-1)
        de_latent2_2 = ut.sample_gaussian(qmu2_2, qvar2_2, args.device)

        dec2_1, mu_dn2_1, var_dn2_1 = self.TCONV2_2.decode(de_latent2_2)
        prec_up2_1 = (var_up2_1 ** (-1)).repeat(class_num, 1)
        prec_dn2_1 = var_dn2_1 ** (-1)
        qmu2_1 = (mu_up2_1.repeat(class_num, 1) * prec_up2_1 + mu_dn2_1 * prec_dn2_1) / (prec_up2_1 + prec_dn2_1)
        qvar2_1 = (prec_up2_1 + prec_dn2_1) ** (-1)
        de_latent2_1 = ut.sample_gaussian(qmu2_1, qvar2_1, args.device)

        dec1_2, mu_dn1_2, var_dn1_2 = self.TCONV2_1.decode(de_latent2_1)
        prec_up1_2 = (var_up1_2 ** (-1)).repeat(class_num, 1)
        prec_dn1_2 = var_dn1_2 ** (-1)
        qmu1_2 = (mu_up1_2.repeat(class_num, 1) * prec_up1_2 + mu_dn1_2 * prec_dn1_2) / (prec_up1_2 + prec_dn1_2)
        qvar1_2 = (prec_up1_2 + prec_dn1_2) ** (-1)
        de_latent1_2 = ut.sample_gaussian(qmu1_2, qvar1_2, args.device)

        dec1_1, mu_dn1_1, var_dn1_1 = self.TCONV1_2.decode(de_latent1_2)
        prec_up1_1 = (var_up1_1 ** (-1)).repeat(class_num, 1)
        prec_dn1_1 = var_dn1_1 ** (-1)
        qmu1_1 = (mu_up1_1.repeat(class_num, 1) * prec_up1_1 + mu_dn1_1 * prec_dn1_1) / (prec_up1_1 + prec_dn1_1)
        qvar1_1 = (prec_up1_1 + prec_dn1_1) ** (-1)
        de_latent1_1 = ut.sample_gaussian(qmu1_1, qvar1_1, args.device)

        x_re = self.TCONV1_1.final_decode(de_latent1_1)

        return x_re.view(bs, class_num, *x.size()[1:])



 