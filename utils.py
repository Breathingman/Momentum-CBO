from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
from Dynamics import getDynamics
from model import Model

class X_Data(Dataset):
    def __init__(self, A):
        self.Data = A

    def __len__(self):
        return np.shape(self.Data)[0]

    def __getitem__(self,idx):
        return self.Data[idx]
    
class Lyapunov_Data(Dataset):
    def __init__(self, A, b):
        self.Data = A
        self.Func = b

    def __len__(self):
        return np.shape(self.Data)[0]

    def __getitem__(self,idx):
        return self.Data[idx], self.Func[idx]


def plot(directory,type,logscale):

    f = open(directory+"\\"+str(type)+".txt","r")
    consensus = []
    i = 0
    while True:
        line = f.readline()
        if not line: break

        line = line.split(" ")[2]
        line = line.replace("Step ","")
        line = line.replace("tensor(","")
        line = line.replace(",","")
        if logscale:
            consensus.append(math.log(abs(float(line))+1e-12))
        else:
            consensus.append((float(line)))
            

    plt.figure()
    plt.plot(consensus)
    plt.xlabel("iteration")
    if logscale:
        plt.ylabel(type+"_logscale")
    else:
        plt.ylabel(type)
    plt.savefig(directory+"\\"+str(type)+".png")


def plotfunc(model, dim, eq, directory, dynamics):

    xr = np.linspace(-1, 1, 201)+eq[0,0]
    yr = np.linspace(-1, 1, 201)+eq[0,1]

    Xm, Ym = np.meshgrid(xr, yr)
    Z = np.zeros((201,201))
    W = np.zeros((201,201))
    V_0 = model.getValue(eq)

    for ix,xv in enumerate(xr):
        for iy,yv in enumerate(yr):

            xyv = torch.tensor([[xv,yv]]).float()
            Z[ix,iy] = model.getValue(xyv)-V_0
            DV_grad = torch.squeeze(model.getGrad(xyv))
            vel = torch.squeeze(getDynamics(xyv,dynamics))
            W[ix,iy] = torch.dot(DV_grad,vel)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(Xm, Ym, Z, cmap='viridis', edgecolor='none')
    ax.plot_wireframe(Xm, Ym, W)

    fig.savefig(directory+"\\"+"V and orbital.png")

    ax.view_init(0,0)
    fig.savefig(directory+"\\"+"V and orbital_fromside.png")

    ax.view_init(0,90)
    fig.savefig(directory+"\\"+"V and orbital_fromotherside.png")

    

