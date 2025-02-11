import torch
import Dynamics

class Model():

    def __init__(self, dim, cpx, grad):

        self.params = []
        self.Quad = torch.rand((dim,dim),dtype = float,requires_grad = grad)*2-1
        self.Coscpx = torch.rand((cpx,dim), dtype = float,requires_grad = grad)*2-1
        self.Cosweight = torch.rand(cpx, dtype = float,requires_grad = grad)*2-1
        self.Sincpx = torch.rand((cpx,dim), dtype = float,requires_grad = grad)*2-1
        self.Sinweight = torch.rand(cpx, dtype = float,requires_grad = grad)*2-1
        self.Logcpx = torch.rand((cpx,dim), dtype = float,requires_grad = grad)*2-1
        self.Logweight = torch.rand(cpx, dtype = float,requires_grad = grad)*2-1

        self.params.append(self.Quad)
        self.params.append(self.Coscpx)
        self.params.append(self.Cosweight)
        self.params.append(self.Sincpx)
        self.params.append(self.Sinweight)
        self.params.append(self.Logcpx)
        self.params.append(self.Logweight)

    def setparams(self, params):

        for i in range(7):
            self.params[i] = params[i]

    def getValue(self, x, mode = "Vanila"):

        x = torch.tensor(x, dtype = float)
        trx = torch.transpose(x,0,1)

        if mode == "Vanila":
            quadv = torch.diagonal(torch.matmul(x,torch.matmul(self.params[0],trx)))
            cosv = torch.matmul(self.params[2],torch.cos(torch.matmul(self.params[1], trx)))
            sinv = torch.matmul(self.params[4],torch.sin(torch.matmul(self.params[3], trx)))
            lnv = torch.matmul(self.params[6],torch.log(torch.abs(torch.matmul(self.params[5], trx)+1e-6)))

        else:

            V_X = torch.zeros(x.shape[0])
            dim = x.shape[-1]
            for i in range(dim):
                V_X = V_X + self.getValue(x[i], "Vanila")

            return V_X

        return (quadv+cosv+sinv)#+lnv/

    def getGrad(self, x, mode = "Vanila"): 

        x = torch.tensor(x, dtype = float)
        trx = torch.transpose(x,0,1)

        if mode == "Vanila":
            grad_quadv = torch.matmul(self.params[0], trx)+torch.matmul(torch.transpose(self.params[0],0,1), trx)
            grad_cosv = -torch.matmul(torch.matmul(torch.transpose(self.params[1],0,1),torch.diag(self.params[2])),torch.sin(torch.matmul(self.params[1], trx)))
            grad_sinv = torch.matmul(torch.matmul(torch.transpose(self.params[3],0,1),torch.diag(self.params[4])),torch.cos(torch.matmul(self.params[3], trx)))
            grad_lnv = torch.matmul(torch.matmul(torch.transpose(self.params[5],0,1),torch.diag(self.params[6])),1/(torch.matmul(self.params[5], trx)+1e-6))       
            
        else:

            dim = x.shape[-1]
            Grad = torch.zeros(x.shape)
            for i in range(dim):
                Grad[i] = self.getGrad(x[i], "Vanila")

        grad = grad_quadv + grad_cosv + grad_sinv#+ grad_lnv
        return torch.transpose(grad,0,1)

    def getL1(self):

        ret = 0
        for (j,param) in enumerate(self.params):
            ret = ret + torch.sum(torch.abs(param))
        
        return ret
    
    def getParamgrad(self, x, eq, alpha, gamma, lambda_, dynamics):

        Paramsgrad = []
        for par in self.params:
            Paramsgrad.append(torch.autograd.grad(self.getLoss(x, eq, alpha, gamma, lambda_, dynamics), par)[0])

        return Paramsgrad
    

    def getLoss(self, x, eq, alpha, gamma, lambda_ = 0, dynamics = "HarmonicOscillator", mode = "Vanila"):

        V_0 = self.getValue(eq, mode)
        V_X = self.getValue(x, mode)
        DV = torch.diagonal(torch.matmul(self.getGrad(x, mode),torch.transpose(Dynamics.getDynamics(x, dynamics = dynamics),0,1)))

        PosDef = torch.relu(V_0-V_X+alpha*torch.sum(torch.square(x),axis = 1))
        NegDef = torch.relu(DV+gamma*torch.sum(torch.square(x), axis = 1))
        #NegDef = torch.heaviside(DV-0.001,torch.zeros(DV.shape, dtype = float))
        #NegDef = torch.abs(DV)

        return torch.mean(torch.sqrt(PosDef+NegDef))+lambda_*self.getL1()


    def getLoss_regression(self, eq, alpha, x, y):

        V_0 = self.getValue(eq)
        V_X = self.getValue(x)
        V_ST = self.getValue(torch.tensor([[1,1]]))

        PosDef = torch.relu(V_0-V_X+alpha*torch.sum(torch.square(x),axis = 1))
        return torch.mean(torch.abs(V_X - V_ST - y)+1e-12)+torch.mean(PosDef)
