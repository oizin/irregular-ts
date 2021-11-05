import numpy as np
import matplotlib.pyplot as plt
import torch

def join_trajectories_gaussian(preds_j):
    """
    One person preds
    """
    mu = []
    sigma = []
    for t in range(len(preds_j)):
        mu.append(np.array([p[0].item() for p in preds_j[t]]))
        sigma.append(np.array([p[1].item() for p in preds_j[t]]))
    return np.concatenate(mu),np.concatenate(sigma)

def join_trajectories_point(preds_j):
    """
    One person preds
    """
    mu = []
    for t in range(len(preds_j)):
        mu.append(np.array([p[0].item() for p in preds_j[t]]))
    return np.concatenate(mu)

def time_trajectories(dt_j,nsteps):
    ts = []
    for t in range(dt_j.size(0)):
        ts.append(np.linspace(dt_j[t][0],dt_j[t][1],nsteps))
    return np.concatenate(ts)

def obs_data(x_j,y_j,dt_j):
    ys = torch.cat((x_j[:,0][0].unsqueeze(0),y_j))
    ts = torch.cat((dt_j.squeeze(0)[0],dt_j.squeeze(0)[1:,1]))
    return ys.numpy(),ts.numpy()

def plot_trajectory_dist(t_j,ys_j,ts_j,mu_tj,sigma_tj,y_full,t_full,ginv=lambda x: x,xlabel="x",ylabel="y",title="",sim=False):
    levels = np.linspace(0.1, 1.96, 5)
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.scatter(t_j,ginv(ys_j),color='r')
    ax.scatter(t_full,y_full,color='r',s=0.1)
    if sim == True:
        ax.plot(t_full,y_full,color='r',linewidth=0.1)
    for level in levels:
        ax.fill_between(ts_j, ginv(mu_tj - level*sigma_tj), 
                        ginv(mu_tj + level*sigma_tj), color='b', alpha=.1, edgecolor='w')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return

def plot_trajectory_point(t_j,ys_j,ts_j,mu_tj,y_full,t_full,ginv=lambda x: x,xlabel="x",ylabel="y"):
    levels = np.linspace(0.1, 1.0, 5)
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.scatter(t_j,ginv(ys_j),color='r')
    ax.plot(t_full,y_full,color='r',linewidth=0.1)
    ax.plot(ts_j,ginv(mu_tj),color='b')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return 