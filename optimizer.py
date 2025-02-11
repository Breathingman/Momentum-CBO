import numpy as np
import math 
import torch
import time


### Parameters for Adam_CB)

def CBO_update(models, x, eq, alpha, gamma, dynamics, optimizer_params, mode = "CBO", Adam_params = ""):

    beta_CBO = optimizer_params["beta_CBO"]
    lambda_CBO = optimizer_params["lambda_CBO"]
    sigma_CBO = optimizer_params["sigma_CBO"]
    learning_rate = optimizer_params["learning_rate"]
    l1reg = optimizer_params["l1reg"]
    N = len(models)

    if mode == "Ad_CBO":
        lambda_CBO1 = optimizer_params["lambda_CBO1"]

    if mode == "Adam_CBO" or mode == "AdamCBO_regression":

        momentum_list = Adam_params.momentum_list
        momentumvar_list = Adam_params.momentumvar_list
        beta_1 = Adam_params.beta_1
        beta_2 = Adam_params.beta_2
        t = Adam_params.t
        t = t+1
        Adam_params.t = t

        if len(momentum_list) == 0:

            for (j,model) in enumerate(models):
                tmp1 = []
                tmp2 = []
                for (l,p) in enumerate(model.params):
                    tmp1.append(torch.zeros(p.shape))
                    tmp2.append(torch.zeros(p.shape))

                momentum_list.append(tmp1)
                momentumvar_list.append(tmp2)

    Lval = np.zeros(N)   
    for (j,model) in enumerate(models):

        if not (mode == "regression" or mode == "AdamCBO_regression"):
            output_Lyapunov = model.getLoss(x, eq, alpha, gamma, l1reg, dynamics)
        else:
            output_Lyapunov = model.getLoss_regression(eq, alpha, x[0], x[1])
        output_Lyapunov = np.squeeze(output_Lyapunov.detach().numpy())
        Lval[j] = np.exp(min(-beta_CBO*np.mean(output_Lyapunov),20))


    alpha_list = Lval/np.sum(Lval)
    sn = []
    mean = []

    current_params = []
    next_params = []
    sn_grad_params = []
    
    for i in range(N):
        current_params.append(models[i].params)

    for (l,p) in enumerate(models[0].params):

        tmp = torch.zeros(p.shape)
        tmp2 = torch.zeros(p.shape)
        for j in range(N):
            tmp = tmp + alpha_list[j]*current_params[j][l]
            tmp2 = tmp2 + current_params[j][l]/N

        sn.append(tmp)
        mean.append(tmp2)

    if mode == "gradguide":
        sn_model = models[0]
        sn_model.setparams(sn)

        sn_grad_params = sn_model.getParamgrad(x, eq, alpha, gamma, l1reg, dynamics)

    for j in range(N):
        params_list = []
        for (l,p) in enumerate(current_params[j]):

            Zrand = torch.randn(current_params[j][l].shape)
            if mode == "gradguide":
                temp_params = temp_params - learning_rate*sn_grad_params[l]

            elif mode == "Adam_CBO" or mode == "AdamCBO_regression":

                momentum_list[j][l] = momentum_list[j][l]*beta_1+(current_params[j][l]-sn[l])*(1-beta_1)
                momentumvar_list[j][l] = momentumvar_list[j][l]*beta_2+(current_params[j][l]-sn[l])*(current_params[j][l]-sn[l])*(1-beta_2)
                temp_params = current_params[j][l] - lambda_CBO*momentum_list[j][l]/(torch.sqrt(momentumvar_list[j][l]/(1-math.pow(beta_2,t)))+1e-12)/(1-math.pow(beta_1,t)) + math.pow(sigma_CBO,t/10)*Zrand
               
            elif mode == "Ad_CBO":
                temp_params = (1-lambda_CBO*learning_rate)*current_params[j][l] + (lambda_CBO+lambda_CBO1)*learning_rate*sn[l] - lambda_CBO1*mean[l] + sigma_CBO*math.sqrt(learning_rate)*(sn[l]-current_params[j][l])*Zrand

            else:
                temp_params = (1-lambda_CBO*learning_rate)*current_params[j][l] + lambda_CBO*learning_rate*sn[l] + sigma_CBO*math.sqrt(learning_rate)*(sn[l]-current_params[j][l])*Zrand
                
            params_list.append(temp_params)
        next_params.append(params_list)

    if not Adam_params == "":
        Adam_params.momentum_list = momentum_list
        Adam_params.momentumvar_list = momentumvar_list

    return next_params

def Grad_update(models, x, eq, alpha, gamma, dynamics, optimizer_params):

    learning_rate = optimizer_params["learning_rate"]
    l1reg = optimizer_params["l1reg"]
    Params_grad = models[0].getParamgrad(x, eq, alpha, gamma, l1reg, dynamics)
    new_params = []
    for (i,param) in enumerate(models[0].params):
        new_params.append(param - learning_rate*Params_grad[i])

    return [new_params]

def Momentum_CBO_update(models, x, eq, alpha, gamma, dynamics, optimizer_params, Adam_params = ""):


    beta_CBO = optimizer_params["beta_CBO"]
    lambda_CBO = optimizer_params["lambda_CBO"]
    sigma_CBO = optimizer_params["sigma_CBO"]
    M_CBO = optimizer_params["M_CBO"]
    learning_rate = optimizer_params["learning_rate"]
    decay_rate = optimizer_params["decay_rate"]
    l1reg = optimizer_params["l1reg"]
    N = len(models)

    
    momentum_list = Adam_params.momentum_list
    beta_1 = Adam_params.beta_1
    t = Adam_params.t
    t = t+1
    Adam_params.t = t

    if len(momentum_list) == 0:

        for (j,model) in enumerate(models):
            tmp1 = []
            for (l,p) in enumerate(model.params):
                tmp1.append(torch.zeros(p.shape))
            momentum_list.append(tmp1)


    Lval = np.zeros(N)   
    for (j,model) in enumerate(models):
        output_Lyapunov = model.getLoss(x, eq, alpha, gamma, l1reg, dynamics)
        output_Lyapunov = np.squeeze(output_Lyapunov.detach().numpy())
        Lval[j] = np.mean(output_Lyapunov)

    current_params = []

    for i in range(N):
        current_params.append(models[i].params)
    next_params = current_params

    rp = np.random.permutation(N)

    for j in range(N//M_CBO):

        batch = rp[j*M_CBO:(j+1)*M_CBO]

        Lval[batch] = np.exp(-beta_CBO*(Lval[batch] - min(Lval[batch])))
        alpha = Lval[batch]/np.sum(Lval[batch])

        sn = []
        for (l,p) in enumerate(models[0].params):
            tmp = torch.zeros(p.shape)
            for (i,k1) in enumerate(batch):
                tmp = tmp + alpha[i]*current_params[k1][l]
            sn.append(tmp)

        for k in batch:
            params_list = []
            for (l,p) in enumerate(current_params[0]):

                Zrand = torch.randn(current_params[k][l].shape)
                momentum_list[k][l] = momentum_list[k][l]*beta_1+(current_params[k][l]-sn[l])*(1-beta_1) 
                if t < 1000:    
                    temp_params = current_params[k][l] - lambda_CBO*momentum_list[k][l] + math.pow(sigma_CBO,t/decay_rate)*Zrand
                else:
                    temp_params = current_params[k][l] - 2*lambda_CBO*momentum_list[k][l] + math.pow(sigma_CBO,t/decay_rate)*Zrand
                   
                params_list.append(temp_params)
            next_params[k] = params_list

    if not Adam_params == "":
        Adam_params.momentum_list = momentum_list

    return next_params


def Adam_update(models, x, eq, alpha, gamma, dynamics, optimizer_params, Adam_params = ""):

    learning_rate = optimizer_params["learning_rate"]
    l1reg = optimizer_params["l1reg"]    
    momentum_list = Adam_params.momentum_list
    momentumvar_list = Adam_params.momentumvar_list
    beta_1 = Adam_params.beta_1
    beta_2 = Adam_params.beta_2
    t = Adam_params.t
    t = t+1
    Adam_params.t = t

    if len(momentum_list) == 0:

        for (l,p) in enumerate(models[0].params):
            momentum_list.append(torch.zeros(p.shape))
            momentumvar_list.append(torch.zeros(p.shape))
      

    Params_grad = models[0].getParamgrad(x, eq, alpha, gamma, l1reg, dynamics)
    new_params = []
    for (i,param) in enumerate(models[0].params):

        momentum_list[i] = momentum_list[i]*beta_1+Params_grad[i]*(1-beta_1)
        momentumvar_list[i] = momentumvar_list[i]*beta_2+Params_grad[i]*Params_grad[i]*(1-beta_2)
        new_params.append(param - learning_rate*momentum_list[i]/(torch.sqrt(momentumvar_list[i]/(1-math.pow(beta_2,t)))+1e-12)/(1-math.pow(beta_1,t)))

    Adam_params.momentum_list = momentum_list
    Adam_params.momentumvar_list = momentumvar_list

    return [new_params]

    
def optimizer_step(models, x, eq, alpha, gamma, dynamics, optimizer_name, optimizer_params, Adam_params = ""):


    updated_params = []
    if optimizer_name == "CBO":    
        updated_params = CBO_update(models, x, eq, alpha, gamma, dynamics, optimizer_params)
    elif optimizer_name == "CBO_Gradguide":
        updated_params = CBO_update(models, x, eq, alpha, gamma, dynamics, optimizer_params, mode = "gradguide")
    elif optimizer_name == "CBO_Regression":
        updated_params = CBO_update(models, x, eq, alpha, gamma, dynamics, optimizer_params, mode = "regression")
    elif optimizer_name == "GradDesc":
        updated_params = Grad_update(models, x, eq, alpha, gamma, dynamics, optimizer_params)
    elif optimizer_name == "ADAM_CBO":
        updated_params = CBO_update(models, x, eq, alpha, gamma, dynamics, optimizer_params, mode = "Adam_CBO", Adam_params=Adam_params)
    elif optimizer_name == "ADAMCBO_Regression":
        updated_params = CBO_update(models, x, eq, alpha, gamma, dynamics, optimizer_params, mode = "AdamCBO_regression", Adam_params=Adam_params)
    elif optimizer_name == "ADAM":
        updated_params = Adam_update(models, x, eq, alpha, gamma, dynamics, optimizer_params, Adam_params = Adam_params)
    elif optimizer_name == "AD_CBO":
        updated_params = CBO_update(models, x, eq, alpha, gamma, dynamics, optimizer_params, mode = "Ad_CBO")
    elif optimizer_name == "Momentum_CBO":
        updated_params = Momentum_CBO_update(models, x, eq, alpha, gamma, dynamics, optimizer_params, Adam_params = Adam_params)
    
    for (l,model) in enumerate(models):
        model.setparams(updated_params[l])
   

