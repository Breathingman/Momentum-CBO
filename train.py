import numpy as np
import os
import torch
import Dynamics
import random
from utils import *
from model import Model
from optimizer import optimizer_step
from Adam_Params import Adam_Params
from torch.utils.data import DataLoader



def train(hyperparams, save_dir = "Results"):

    CBO_list = ["CBO","CBO_Gradguide", "CBO_Regression", "ADAM_CBO", "ADAMCBO_Regression", "AD_CBO", "Momentum_CBO"]
    Grad_list = ["GradDesc", "ADAM"]

    optimizer_name = hyperparams["Opt_scheme"]
    num_data = hyperparams["num_data"]
    dim = hyperparams["dim"]
    batch_size = hyperparams["batch_size"]
    epochs = hyperparams["epochs"]
    alpha = hyperparams["alpha"]
    gamma = hyperparams["gamma"]
    dynamics = hyperparams["dynamics"]
    cpx = hyperparams["sums"]
    store_steps = hyperparams["store_steps"]

    X = np.zeros((num_data,dim))
    Y = np.zeros(num_data)

    if optimizer_name == "CBO_Regression":

        data_directory = hyperparams["regress_datadir"]
        with open(data_directory, 'r') as f:

            strings = f.readlines()
            for i in range(num_data):

                line = strings[i]
                line = line.split()

                X[i][0] = float(line[0])
                X[i][1] = float(line[1])
                Y[i] = line[2]
        
        f.close()
    else:
        
        input_range = hyperparams["input_range"]
        X = (np.random.rand(num_data,dim)*2-1)*input_range

    if optimizer_name in CBO_list:

        Models = []
        Losses = []

        N = hyperparams["CBO_params"]["N_CBO"]
        l1reg = hyperparams["CBO_params"]["l1reg"]
        CBO_batch = hyperparams["CBO_params"]["M_CBO"]
        
        eq = Dynamics.getEq(dynamics, dim)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if optimizer_name == "CBO_Regression" or optimizer_name == "ADAMCBO_Regression":
            training_data = Lyapunov_Data(X+eq, Y)
        else:
            training_data = X_Data(X+eq)    

        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

        for _ in range(N):
            new_model = Model(dim,cpx,False)
            Models.append(new_model)

        if optimizer_name == "ADAM_CBO" or optimizer_name == "ADAMCBO_Regression" or optimizer_name == "Momentum_CBO":
            Adam_params = Adam_Params()
        else:
            Adam_params = ""

        for i in range(epochs):
            for _, samples in enumerate(train_dataloader):
                
                f = open(save_dir+"\\losses.txt","a")

                if optimizer_name == "CBO_Regression" or optimizer_name == "ADAMCBO_Regression":
                      
                    batch_x, batch_y = samples

                    if optimizer_name == "CBO_Regression":
                        optimizer_step(Models, [batch_x, batch_y], eq, alpha, gamma, dynamics, optimizer_name = optimizer_name, optimizer_params = hyperparams["CBO_params"]) 
                    else:
                        optimizer_step(Models, [batch_x, batch_y], eq, alpha, gamma, dynamics, optimizer_name = optimizer_name, optimizer_params = hyperparams["CBO_params"], Adam_params = Adam_params) 
                    print("Step "+str(i)+": "+str(Models[0].getLoss_regression(eq, alpha, batch_x, batch_y)))
                    f.write("Step "+str(i)+": "+str(Models[0].getLoss_regression(eq, alpha, batch_x, batch_y)))
                    f.write("\n")

                else:

                    batch_x = samples
                    optimizer_step(Models, samples, eq, alpha, gamma, dynamics, optimizer_name = optimizer_name, optimizer_params = hyperparams["CBO_params"], Adam_params = Adam_params) 
                    print("Step "+str(i)+": "+str(Models[0].getLoss(samples, eq, alpha, gamma, l1reg, dynamics)))
                    f.write("Step "+str(i)+": "+str(Models[0].getLoss(samples, eq, alpha, gamma, l1reg, dynamics)))
                    f.write("\n")
                    
                f.close()

                f = open(save_dir+"\\consensus.txt","a")
                list_ = Models[0].params
                l1sum = 0
                for (j,_) in enumerate(list_):
                    if j<5:
                        l1sum = l1sum + torch.sum(Models[0].params[j]-Models[1].params[j])
                f.write("Step "+str(i)+": "+str(abs(l1sum)))
                f.write("\n")
                f.close()

                if (i+1)%store_steps == 0:
                    f = open(save_dir+"\\iteration "+str(i)+".txt","w")
                    for j in range(10):
                        f.write("Model "+str(j)+":\n")
                        f.write(str(Models[j].params))
                        f.write("\n")
                    f.close()

        plot(save_dir,"consensus",True)
        plot(save_dir,"losses",False)
        plotfunc(Models[0],dim,eq,save_dir, dynamics)

    elif optimizer_name in Grad_list:

        l1reg = hyperparams["GD_params"]["l1reg"]
        eq = Dynamics.getEq(dynamics, dim)

        if optimizer_name == "ADAM":
            Adam_params = Adam_Params()
        else:
            Adam_params = ""

        training_data = X_Data(X)
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

        Models = [Model(dim, cpx, True)]
        for i in range(epochs):
            for _, samples in enumerate(train_dataloader):

                batch_x = samples
                #optimizer_step(Models, [batch_x, batch_y], eq, alpha, gamma, dynamics, optimizer_name = optimizer_name, optimizer_params = hyperparams["GD_params"])
                
                optimizer_step(Models, samples, eq, alpha, gamma, dynamics, optimizer_name = optimizer_name, optimizer_params = hyperparams["GD_params"], Adam_params = Adam_params)
                print("Step "+str(i)+": "+str(Models[0].getLoss(batch_x, eq, alpha, gamma, l1reg, dynamics)))

                f = open(save_dir+"\\losses.txt","a")
                f.write("Step "+str(i)+": "+str(Models[0].getLoss(batch_x, eq, alpha, gamma, l1reg, dynamics)))
                f.write("\n")
                f.close()
                
                if (i+1)%store_steps == 0:
                    f = open(save_dir+"\\iteration "+str(i)+".txt","w")
                    f.write(str(Models[0].params))
                    f.close()

        plot(save_dir,"losses",False)
        plotfunc(Models[0],dim,eq,save_dir, dynamics)