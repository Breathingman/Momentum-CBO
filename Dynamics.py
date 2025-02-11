import torch
import numpy as np

eq = np.array([[0,0.0]])

def getDynamics(x, dynamics):

    y = torch.zeros(x.shape, dtype = float)

    if dynamics == "Sink":

        y[:,0] = -x[:,0]
        y[:,1] = -x[:,1]

    elif dynamics == "DampedHarmonicOscillator":

        y[:,0] = x[:,1]
        y[:,1] = -x[:,0]-x[:,1]

    elif dynamics == "DampedPendulum":

        y[:,0] = x[:,1]
        y[:,1] = -x[:,1]-torch.sin(x[:,0])

    elif dynamics == "Lorenz":

        y[:,0] = 3*(-x[:,0]+x[:,1])
        y[:,1] = 0.5*x[:,0]-x[:,1]-x[:,0]*x[:,2]
        y[:,2] = -x[:,2]+x[:,0]*x[:,1]

    elif dynamics == "HarmonicOscillator":

        y[:,0] = -10*x[:,1]-7*x[:,0]
        y[:,1] = 5*x[:,0] + 7*x[:,1]

    elif dynamics == "Pendulum":

        y[:,0] = x[:,1]
        y[:,1] = -torch.sin(x[:,0])

    elif dynamics == "VanDerPolPerturb":

        y[:,0] = x[:,1]
        y[:,1] = -x[:,0]+0.5*(x[:,1]*x[:,1]*x[:,1]/3-x[:,1])

    elif dynamics == "Swirl":

        y[:,0] = torch.sin(x[:,1])*torch.cos(x[:,0])-torch.cos(x[:,1])*torch.sin(x[:,0])
        y[:,1] = -torch.sin(x[:,1])*torch.cos(x[:,0])-torch.cos(x[:,1])*torch.sin(x[:,0])

    elif dynamics == "Winfree":

        y[:,0] = 1 - (1+torch.cos(x[:,1]))*torch.sin(x[:,0])
        y[:,1] = 1 - (1+torch.cos(x[:,0]))*torch.sin(x[:,1])
 
    return y

def getEq(dynamics, dim):

    if dim==3:
        return np.array([[0,0,0.0]])
    elif dynamics == "LotkaVolterra":
        return np.array([[1,1.0]])
    elif dynamics == "Winfree":
        return np.array([[0.57482, 0.57482]])
    else:
        return np.array([[0.0,0]])