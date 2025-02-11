from model import Model
import torch
import numpy as np
import Dynamics
from utils import plot,plotfunc

a = Model(2,2,True)
a.params =	[torch.tensor([[45.7918, 47.0624],
        [58.3639, 58.3477]], dtype=torch.float64), torch.tensor([[0.5189, 2.7339],
        [5.2906, 7.3924]], dtype=torch.float64), torch.tensor([-1.8841, -2.9312], dtype=torch.float64), torch.tensor([[5.1011, 7.3887],
        [2.8343, 3.7152]], dtype=torch.float64), torch.tensor([-0.0317,  0.5983], dtype=torch.float64), torch.tensor([[11.5978, -5.1531],
        [-1.1758,  1.4650]], dtype=torch.float64), torch.tensor([-9.1424,  8.3414], dtype=torch.float64)]
#
#plot("Results\\Momentum_CBO\\2.Invariant\\Pendulum_1","consensus",True)
#plot("Results_final\\Momentum\\Swirl","losses",True)
plotfunc(a,2,np.array([[0,0]]),"Results_final\\Momentum", "HarmonicOscillator")