# 
from load_data import *
from torchvision.utils import make_grid
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T

import torchvision.transforms.functional as F

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


trainset, valset, testset_seen, testset_unseen, channel, seen_classes = get_dataset('CIFAR10', 777, 10, 6, 117)
loader = data.DataLoader(trainset, batch_size=5, shuffle=False, drop_last=False, num_workers=1, pin_memory=True)
# transform = T.Compose([T.ColorJitter(0.4, 0.4, 0.4, 0.1), T.RandomGrayscale(p=0.3)])
transform = T.Compose([T.ColorJitter(0, 0, 0, 0), T.RandomGrayscale(p=0)])


mean = np.array([0.4914, 0.4822, 0.4465])
std = np.array([0.2023, 0.1994, 0.2010])

invTrans = T.Compose([ T.Normalize(mean = [ 0., 0., 0. ],
                                    std = 1/std),
                        T.Normalize(mean = -mean,
                                    std = [ 1., 1., 1. ]),])

                            
for img, lab in loader: 
    img = invTrans(img)
    imgs = torch.cat((img, transform(img)), 0)
    grid = make_grid(imgs, nrow=5, ncol=2)
    plt.imshow(grid.permute(1,2,0))
    plt.show()

