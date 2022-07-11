from itertools import product
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import imageio
import random as rd
import time
import torch
import os
# import indicator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def LearningRateScheduler(loss_his, optimizer, lr_base):

    loss_his = np.array(loss_his)
    num_shock = np.sum((loss_his[:-1] - loss_his[1:]) < 0)
    if num_shock > 0.40 * loss_his.shape[0] and lr_base > 1e-4:
        lr_new = lr_base * 0.8
        adjust_learning_rate(optimizer, lr_new)
    else:
        lr_new = lr_base
    print('*** lr {} -> {}  ***'.format(lr_base, lr_new))
    print('num_shock', num_shock)
    return lr_new


def SaveData(input, latent, label, dist=None, path='', name=''):

    if type(input) == torch.Tensor:
        input = input.detach().cpu().numpy()
    if type(latent) == torch.Tensor:
        latent = latent.detach().cpu().numpy()
    if type(label) == torch.Tensor:
        label = label.detach().cpu().numpy()

    numEpoch = int(name.split('train_epoch')[1])
    np.save(
        path + name + 'latent.npy',
        latent,
    )

    if numEpoch < 1:
        np.save(
            path + name + 'input.npy',
            input.astype(np.float16),
        )
        np.save(
            path + name + 'label.npy',
            label.astype(np.float16),
        )
        if dist is not None:
            np.save(
                path + name + 'dist.npy',
                dist.detach().cpu().numpy().astype(np.float16),
            )


class GIFPloter():
    def __init__(self, ):
        self.path_list = []

    def PlotOtherLayer(
        self,
        fig,
        data,
        label,
        title='',
        fig_position0=1,
        fig_position1=1,
        fig_position2=1,
        s=0.1,
        graph=None,
        link=None,
        #    latent=None,
    ):
        from sklearn.decomposition import PCA

        color_list = []
        for i in range(label.shape[0]):
            color_list.append(int(label[i]))

        if data.shape[1] > 3:
            pca = PCA(n_components=2)
            data_em = pca.fit_transform(data)
        else:
            data_em = data

        # data_em = data_em-data_em.mean(axis=0)

        if data_em.shape[1] == 3:
            ax = fig.add_subplot(fig_position0,
                                 fig_position1,
                                 fig_position2,
                                 projection='3d')

            ax.scatter(data_em[:, 0],
                       data_em[:, 1],
                       data_em[:, 2],
                       c=color_list,
                       s=s,
                       marker='.',
                       cmap='rainbow',
                       )

        if data_em.shape[1] == 2:
            ax = fig.add_subplot(fig_position0, fig_position1, fig_position2)

            if graph is not None:
                self.PlotGraph(data, graph, link)

            s = ax.scatter(data_em[:, 0],
                           data_em[:, 1],
                           c=label,
                           s=s,
                           )
            plt.axis('equal')
            # plt.colorbar(s)
            if None:
                list_i_n = len(set(label.tolist()))
                # print(list_i_n)
                legend1 = ax.legend(*s.legend_elements(num=list_i_n),
                                    loc="upper left",
                                    title="Ranking")
                ax.add_artist(legend1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.xticks([])
        plt.yticks([])
        # plt.title(title)

    def AddNewFig(self,
                  latent,
                  label,
                  link=None,
                  graph=None,
                  his_loss=None,
                  title_='',
                  path='./',
                  dataset=None):

        # fig = plt.figure(figsize=(20, 20))
        fig = plt.figure(figsize=(9, 9))

        if latent.shape[0] <= 1000:
            s=3
        elif latent.shape[0] <= 10000:
            s = 1
        else:
            s = 0.1

        # if latent.shape[1] <= 3:
        self.PlotOtherLayer(fig,
                            latent,
                            label,
                            title=title_,
                            fig_position0=1,
                            fig_position1=1,
                            fig_position2=1,
                            graph=graph,
                            link=link,
                            s=s)
        plt.tight_layout()
        path_c = path + title_

        self.path_list.append(path_c)

        plt.savefig(path_c, dpi=200)
        plt.close()
        import joblib
        joblib.dump(
            [
                latent,
                label,
            ], path_c+'.gz')

    def PlotGraph(self, latent, graph, link):

        for i in range(graph.shape[0]):
            for j in range(graph.shape[0]):
                if graph[i, j] == True:
                    p1 = latent[i]
                    p2 = latent[j]
                    lik = link[i, j]
                    plt.plot([p1[0], p2[0]], [p1[1], p2[1]],
                             'gray',
                             lw=1 / lik)
                    if lik > link.min() * 1.01:
                        plt.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2,
                                 str(lik)[:4],
                                 fontsize=5)

    def SaveGIF(self):

        gif_images = []
        for i, path_ in enumerate(self.path_list):
            # print(path_)
            gif_images.append(imageio.imread(path_))
            # if i > 0 and i < len(self.path_list)-2:
            #     os.remove(path_)

        imageio.mimsave(path_[:-4] + ".gif", gif_images, fps=5)


def SetSeed(seed):
    """function used to set a random seed

    Arguments:
        seed {int} -- seed number, will set to torch, random and numpy
    """
    SEED = seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    rd.seed(SEED)
    np.random.seed(SEED)


def GetPath(name=''):

    rest = time.strftime("%Y%m%d%H%M%S_", time.localtime()) + \
        os.popen('git rev-parse HEAD').read()
    path = 'log/' + rest[:20] + name
    if not os.path.exists(path):
        os.makedirs(path)

    return path + '/'


def SaveParam(path, param):
    import json
    paramDict = param
    paramStr = json.dumps(paramDict, indent=4)
    # paramStr = json.dumps(paramStr)

    print(paramStr)
    print(paramStr, file=open(path + '/param.txt', 'a'))


def ModelSaver(model, path, name):
    torch.save(model.state_dict(), path + name + '.model')


def ModelLoader(model, path, name):
    model.load_state_dict(torch.load(path + name + '.model'))


class AutoTrainer():
    def __init__(self,
                 changeList,
                 paramName,
                 mainFunc,
                 deviceList=[4, 5, 6, 7],
                 poolNumber=4,
                 name='AutoTrainer',
                 waittime=1):
        self.paramName = paramName
        self.mainFunc = mainFunc
        self.changeList = changeList
        self.deviceList = deviceList
        self.poolNumber = poolNumber
        self.name = name
        self.waittime = waittime

        self.loopList = list(product(*tuple(changeList)))
        # print(self.loopList)
        # input()

    def Run(self, ):

        poolLeftNumber = self.poolNumber - 1
        # gpunum = 0
        for i, item in enumerate(self.loopList):

            gpu_index = self.deviceList[i % len(self.deviceList)]
            txtDevice = "CUDA_VISIBLE_DEVICES={} ".format(gpu_index)
            txtmain = 'python -u ' + self.mainFunc #+ ' --gpu_index {} '.format(gpu_index)
            txtparam = ''
            for j, param in enumerate(self.paramName):
                txtparam += '--{} {} '.format(param, item[j])
            txtname = '--name ' + self.name + '{}/{}'.format(i,len(self.loopList))

            txt = ' '.join([txtDevice, txtmain, txtparam, txtname])
            txtp = ' '.join([txtDevice, 'nohup', txtmain, txtparam, txtname])
            txtp = ' '.join([txtDevice, txtmain, txtparam, txtname])
            
            # print(txtp+' &',)
            print(txtp)
            # input()
            # os.system(txt)

            if poolLeftNumber == 0:
                print('continue left:', poolLeftNumber)
                poolLeftNumber = self.poolNumber - 1
                child = subprocess.Popen(txt, shell=True)
                child.wait()
                # subprocess.Popen()
            else:
                print('wait left:', poolLeftNumber)
                child = subprocess.Popen(txt, shell=True)
                # child.wait(2)
                poolLeftNumber -= 1
            time.sleep(self.waittime)




def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)