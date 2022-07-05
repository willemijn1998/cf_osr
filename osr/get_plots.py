import numpy as np
import torchvision.transforms as T
from torchvision.utils import make_grid
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torch.nn.functional as nnF
from babypots import Babyplot 
from sklearn.manifold import TSNE
from umap import UMAP 
from sklearn import decomposition



def plot_rec(model, loader, transform, args, N_plots=5,
                mean=np.array([0.4914, 0.4822, 0.4465]), 
                std = np.array([0.2023, 0.1994, 0.2010])): 

    invTrans = T.Compose([ T.Normalize(mean = [ 0., 0., 0. ],
                                std = 1/std),
                    T.Normalize(mean = -mean,
                                 std = [ 1., 1., 1. ]),])

    for i, (img, lab) in enumerate(loader): 
        labels_onehot = nnF.one_hot(lab, num_classes = args.num_classes).type(torch.float)
        y_latent_mu, predict_test, img_rec = model.test(img, labels_onehot,args)
        imgs = torch.cat((invTrans(img), invTrans(img_rec)), 0)
        grid = make_grid(imgs, nrow=5, ncol=2)
        plt.imshow(grid.permute(1,2,0))
        plt.show()
        if i == N_plots: 
            break 

def get_tsne(feature_path, n_components, n_points): 
    feas = np.load_txt(feature_path)[:n_points]

    tsne = TSNE(n_components=n_components, random_state=0)
    projections = tsne.fit_transform(feas)

    return projections 

def get_umap(feature_path, n_components, n_points): 
    feas = np.load_txt(feature_path)[:n_points]

    umap = UMAP(n_components=n_components, random_state=0)
    projections = umap.fit_transform(feas)

    return projections 

def get_pca(feature_path, n_components, n_points): 
    feas = np.load_txt(feature_path)[:n_points]

    pca = decomposition.PCA(n_components=n_components)
    pca.fit(feas)
    projections = pca.transform(feas)

    return projections 


def point_cloud(projections, target_path, n): 

    bp = Babyplot(background_color="#ffffddff", turntable=True)
       





# def plot_cf(model, loader, args)






 