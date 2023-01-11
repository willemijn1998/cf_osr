import numpy as np
import torchvision.transforms as T
from torchvision.utils import make_grid
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torch.nn.functional as nnF
from babyplots import Babyplot 
from sklearn.manifold import TSNE
# from umap import UMAP 
from sklearn import decomposition
import matplotlib.pyplot as plt
from numpy import genfromtxt
import pandas as pd
import utils as ut
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
numpy2ri.activate()
mvt = importr('mvtnorm')



def plot_rec(model, loader, args, N_plots=5,
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
        plt.xticks([])
        plt.yticks([])
        plt.savefig('images/x_rec/c10_{}.png'.format(i))
        plt.show()
        if i == N_plots: 
            break 

def get_sample_feas(loader_seen, loader_unseen, model, args): 

    sample_feas = []
    sample_tars = []
    with torch.no_grad(): 
        for data, target in loader_seen:
            z_mu, z_var = model.generate_sample_feas(data, args)
            z_mu, z_var = z_mu.to(args.device), z_var.to(args.device)
            z = ut.sample_gaussian(z_mu, z_var, args.device)
            sample_feas.append(z)
            sample_tars.append(target)

        for data, target in loader_unseen:
            z_mu, z_var = model.generate_sample_feas(data, args)
            z_mu, z_var = z_mu.to(args.device), z_var.to(args.device)
            z = ut.sample_gaussian(z_mu, z_var, args.device)
            sample_feas.append(z)
            sample_tars.append(target)

    sample_feas = torch.cat(sample_feas, dim=0)
    sample_tars = torch.cat(sample_tars)

    return sample_feas, sample_tars

def get_hierarchy_feas(loader_seen, loader_unseen, model, args): 
    "Returns a dict with latent features of every level in LVAE"

    hier_feas = {i: [] for i in range(10)}
    targets = []
    with torch.no_grad(): 
        for data, target in loader_seen: 
            data, target = data.to(args.device), target.to(args.device)
            target_onehot = nnF.one_hot(target, num_classes = args.num_classes).type(torch.float)
            target_onehot = target_onehot.to(args.device)
            features = model.latent_hierarchy(data, target_onehot, args)
            targets += [target]

            for i, fea_level in enumerate(features):
                hier_feas[i] += [fea_level]  

        for data, target in loader_unseen: 
            data, target = data.to(args.device), target.to(args.device)
            # target_onehot = nnF.one_hot(target, num_classes = args.num_classes).type(torch.float)
            features = model.latent_hierarchy(data, target_onehot, args)
            targets += [target]

            for i, fea_level in enumerate(features):
                hier_feas[i] += [fea_level]  

    for i in range(10): 
        hier_feas[i] = torch.cat(hier_feas[i], dim=0)

    targets = torch.cat(targets)

    return targets, hier_feas   


def get_tsne(feas, n_components): 

    tsne = TSNE(n_components=n_components)
    projections = tsne.fit_transform(feas)

    return projections

def get_umap(save_path, n_components, n_points): 
    feas = np.loadtxt((save_path + "/test_fea.txt"))
    targets = np.loadtxt((save_path + "/test_tar.txt"))

    umap = UMAP(n_components=n_components, random_state=0)
    projections = umap.fit_transform(feas)

    return projections, targets

def get_pca(save_path, n_components, n_points): 

    feas = np.loadtxt((save_path + "/test_fea.txt"))
    targets = np.loadtxt((save_path + "/test_tar.txt"))

    pca = decomposition.PCA(n_components=n_components)
    pca.fit(feas)
    projections = pca.transform(feas)

    return projections, targets 


def point_cloud(projections, targets, save_path): 

    bp = Babyplot(background_color="#ffffddff")
    bp.add_plot(projections.tolist(), "pointCloud", "categories", targets.tolist(), {"colorScale": "Dark2", "size": 5})  
    bp.plots[0]["options"]["showLegend"] = True
    bp.plots[0]["options"]["showAxes"] = [True, True, True]
    bp.save_as_html((save_path+'/pointcloud.html'))

def get_plot_name(args): 
    loss_list = []
    name = "Plot for {} with "
    if args.contrastive_loss: 
        loss_list += ["Contrastive Loss"]
    
    if args.mmd_loss: 
        loss_list += ["MMD Loss"]
    
    if args.supcon_loss: 
        loss_list+=["Supervised Contrastive Loss"]



def scatter_plot(projections, targets, plot_name, means): 
    # fig, ax = plt.subplots()
    # n = int(np.max(targets))
    # scatter = ax.scatter(projections[:-n,0], projections[:-n,1], c=targets[:-n], cmap='Spectral', s=2)
    # if plot_means: 
    #     breakpoint()
    #     ax.scatter(projections[-n:,0], projections[-n:,1], c=targets[-n:], marker=(5,2))
    # legend = ax.legend(*scatter.legend_elements(), loc= "lower left", title="Classes")
    # ax.add_artist(legend)
    # plt.title(plot_name)
    # plt.savefig('images/%s.png' %(plot_name))
    # plt.show()

    colors = ["gold", "limegreen", "lightcoral", "cornflowerblue", "black", "orange", "olive"]
    colors2 = ['darkgoldenrod', 'darkgreen', 'orangered', "midnightblue", "darkslategrey", "darkorange"]

    # no_to_class = {0: "plane", 1: "car", 2: "ship", 3: "truck", 4: "unseen"}

    fig, ax = plt.subplots()
    x, y = projections[:,0], projections[:,1]
    for c in range(len(means)+1):
        ax.scatter(x[targets==c], y[targets==c], c=colors[c], label=c,
                alpha=1, s=3, edgecolors='none')
        # if c == 6: 
        #     break
    for c in range(len(means)): 
        ax.scatter(means[c,0], means[c,1], c=colors2[c], marker=(5,2))

    # ax.legend(loc='upper right')
    ax.grid(True)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    # plt.title("tSNE scatter plot")
    plt.savefig('images/%s.png' %(plot_name))


    plt.show()



# def scatter_plot(projections, targets, plot_name, means): 

#     colors = ["gold", "limegreen", "lightcoral", "cornflowerblue", "black"]
#     colors2 = ['darkgoldenrod', 'darkgreen', 'orangered', "midnightblue", "darkslategrey"]



    

#     fig, ax = plt.subplots()
#     x, y = projections[:,0], projections[:,1]
#     for c in range(len(means)+1):
#         ax.scatter(x[targets==c], y[targets==c], c=colors[c], label=c,
#                 alpha=1, s=3, edgecolors='none')
#         # if c == 6: 
#         #     break
#     for c in range(len(means)): 
#         ax.scatter(means[c,0], means[c,1], c=colors2[c], marker=(5,2))

#     ax.legend()
#     ax.grid(True)
#     ax.axes.xaxis.set_ticklabels([])
#     ax.axes.yaxis.set_ticklabels([])
#     # plt.title("tSNE scatter plot")
#     plt.savefig('images/%s.png' %(plot_name))


#     plt.show()
        

def get_class_means(train_feas, train_tars, n): 
    means = [] 
    for i in range(n): 
        mean = np.mean(train_feas[train_tars ==i], axis=0)
        means.append(mean)
    
    means = np.vstack(means)
    return means

def barchart(file_name='results/CIFAR10_results.csv'): 
    
    df = pd.read_csv(file_name, header=None, names=['labels', 'mean', 'std'])
    labels= df['labels'].tolist()
    means = df['mean'].tolist()
    std = df['std'].tolist()
    labels=['Baseline', 'RecContra', 'RecContra MMD Aug', 'RecContra MMD',
                'MMD Aug', 'MMD', 'RecContra SupCon Aug', 'RecContra Supcon', 
                'SupCon Aug', 'Supcon']    
    plt.bar(labels, means, yerr=std)
    plt.ylim(0.5, 0.8)
    # Rotation of the bars names
    plt.xticks(rotation=90, fontsize=10)
    plt.ylabel('F1 score')
    plt.title('F1 scores for CIFAR10')

    plt.savefig('images/CIFARA10_chart.png', bbox_inches='tight')
    plt.show()

def barchart2(file_name='results/CIFAR10_results.csv'): 
    labels=['Baseline', 'RecConttargetsra', 'RecContra MMD Aug', 'RecContra MMD',
                'MMD Aug', 'MMD', 'RecContra SupCon Aug', 'RecContra Supcon', 
                'SupCon Aug', 'Supcon']  
    df = pd.read_csv(file_name, header=None, names=['labels', 'mean', 'std'])
    df['labels'] = labels 
    df = df.sort_values(by='mean', ascending=False)

    labels= df['labels'][0:5].tolist()
    means = df['mean'][0:5].tolist()
    std = df['std'][0:5].tolist()

    plt.bar(labels, means, yerr=std)
    plt.ylim(0.68, 0.74)
    # Rotation of the bars names
    plt.xticks(rotation=90, fontsize=10)
    plt.ylabel('F1 score')
    plt.title('F1 scores for CIFARA10')

    plt.savefig('images/CIFAR10_chart2.png', bbox_inches='tight')
    plt.show()

# barchart2()
# barchart()

def show_imgs(loader, mean, std, s, p): 
    transform =  T.Compose([T.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s), T.RandomGrayscale(p=p)])	

    invTrans = T.Compose([ T.Normalize(mean = [ 0., 0., 0. ],
                                    std = 1/std),
                        T.Normalize(mean = -mean,
                                    std = [ 1., 1., 1. ]),])
    for img, lab in loader: 
        img = invTrans(img)
        imgs = torch.cat((img, transform(img)), 0)
        grid = make_grid(imgs, nrow=5, ncol=2)
        print(lab)
        plt.imshow(grid.permute(1,2,0))
        plt.show()

def rec_histogram(recs, tars, num_classes, exp_name): 
    recs_seen = recs[tars != num_classes]
    recs_unseen = recs[tars == num_classes]
    bins = np.linspace(np.min(recs), np.max(recs), 100)

    plt.hist(recs_seen, bins, alpha=0.5, label='seen')
    plt.hist(recs_unseen, bins, alpha=0.5, label='unseen')
    plt.legend()
    plt.title(exp_name)
    plot_path = exp_name[:-2] + exp_name[-1]
    plt.savefig('images/rec_hist/%s.png' %(plot_path))
    plt.show()

def get_mu_sigma(testtar, testfea, num_classes): 
    mu = []
    sigma = []
    for i in range(num_classes): 
        m = np.mean(testfea[testtar==i], axis=0)
        s = np.cov((testfea[testtar==i] - m).T)
        mu.append(m) 
        sigma.append(s)    
    return mu, sigma 

def multivariateGaussian(vector, mu, sigma):
    vector = np.array(vector)
    dim = len(vector)
    d = (np.mat(vector - mu)) * np.mat(np.linalg.pinv(sigma)) * (np.mat(vector - mu).T)
    p = np.exp(d * (-0.5)) / (((2 * np.pi) ** int(dim/2)) * (np.linalg.det(sigma)) ** (0.5))
    p = np.log(p)
    return p

def ll_histogram(testtar, testfea, num_classes, mu, sigma): 
    n, dim = testfea.shape
    ll_all = np.zeros(n)
    for i in range(n): 
        max_ll = 0
        for c in range(num_classes): 
            ll = multivariateGaussian(testfea[i], mu[c], sigma[c])
            if ll > max_ll: 
                max_ll = ll
        ll_all[i] = max_ll 

    # return vector with known likelihood scores 
    ll_known = ll_all[testtar < num_classes]
    ll_unknown = ll_all[testtar == num_classes]

    return ll_known, ll_unknown 

def show_recon(loader, model, args): 

    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])

    invTrans = T.Compose([ T.Normalize(mean = [ 0., 0., 0. ],
                                        std = 1/std),
                            T.Normalize(mean = -mean,
                                        std = [ 1., 1., 1. ]),])

                                
    for img, lab in loader: 
        lab_onehot = nnF.one_hot(lab, num_classes = args.num_classes).type(torch.float)
        img = invTrans(img)
        _, _, img_rec = model.test(img, lab_onehot, args)
        imgs = torch.cat((img, img_rec), 0)

        grid = make_grid(imgs, nrow=5, ncol=2)
        plt.imshow(grid.permute(1,2,0))
        plt.savefig('images/rec/%s.png'%args.exp_name[:-2])
        plt.show()





