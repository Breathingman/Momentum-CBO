import numpy as np
import math 
import torch

class Adam_Params:

    def __init__(self):
        self.momentum_list = []
        self.momentumvar_list = []
        self.beta_1 = 0.99
        self.beta_2 = 0.999
        self.t = 0