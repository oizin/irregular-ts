import torch 
import torch.nn as nn
import torch.nn.functional as F
#from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint
import math
from tqdm import tqdm

DT_SCALER = 1 / 100
SEQUENCE_LENGTH = 100

class Baseline(nn.Module):
    """
    Base class for the models
    """
    def __init__(self, input_dim, hidden_dim, p, output_dim, device):
        super(Baseline, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.p = p
        self.device = device
        
    def train_single_epoch(self,dataloader,optim):
        """
        Method for model training
        """
        loss = 0.0
        n_batches = len(dataloader)
        print("number of batchs: {}".format(n_batches))
        for i, (x, y, msk, dt, seqlen) in enumerate(dataloader):
            x = x.to(self.device)
            y = y.to(self.device)
            dt = dt.to(self.device)
            msk = msk.bool().to(self.device)
            optim.zero_grad()
            preds = self.forward(dt,x).squeeze(2)
            loss_step = self.loss_fn(preds,y,~msk.squeeze(0))
            loss_step.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
            optim.step()
            loss += loss_step.item()
            if i % int(n_batches/4) == 0:
                print("Batch number: {}".format(i))
                print("BATCH_loss : {:05.3f}".format(loss_step.item()))
        loss /= (i + 1)
        print("EPOCH_loss : {:05.3f}".format(loss))
        
        return loss
        
    def evaluate(self,dataloader,p=0.0):
        """
        Method for model evaluation
        """
        rmse, loss = 0., 0.
        N = 0
        y_preds = []
        y_tests = []
        msks = []
        #dts = []
        with tqdm(total=len(dataloader)) as t:
            for i, (x, y, msk, dt,seqlen) in enumerate(dataloader):
                N += sum(sum(msk == 0)).item()
                x = x.to(self.device)
                y = y.to(self.device)
                dt = dt.to(self.device)
                # model prediction
                y_ = self.forward(dt,x).squeeze(2)
                y_preds.append([yc.detach().cpu().numpy() for yc in y_]) 
                y_tests.append(y.cpu().numpy())
                msk = msk.bool().to(self.device)
                rmse += self.get_sse(y_,y,~msk.squeeze(0)).item()
                loss += self.loss_fn(y_,y,~msk.squeeze(0)).item()
                msks.append(msk.cpu().numpy())
                t.update()
        rmse /= N
        loss /= (i + 1)
        rmse = math.sqrt(rmse)
        print("_rmse : {:05.3f}".format(rmse))
        print("_loss : {:05.3f}".format(loss))
        return loss,rmse, y_preds, y_tests, msks

    def get_sse(self,y_,y,msk):
        """
        Method for calculation of the sum of squared errors
        """
        if type(y_) == tuple:
            y_ = y_[0]
        c = torch.log(torch.tensor(140.0))
        rmse = torch.sum((torch.exp(y_[msk] + c) - torch.exp(y[msk] + c))**2)
        return rmse

#-------------------------------------------------------------------------------------------------

class NeuralODE_(nn.Module):
    """
    dglucose/dt = NN(glucose,insulin)
    """
    def __init__(self,input_dim,batch_size,device):
        super(NeuralODE_, self).__init__()
        
        self.x = torch.zeros(batch_size,SEQUENCE_LENGTH,input_dim).to(device)
        self.dt = torch.zeros(batch_size,SEQUENCE_LENGTH,1).to(device)
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.Tanh(),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, z):
        xz = torch.cat((z,self.x),2)
        return self.net(xz)*(self.dt*DT_SCALER) # -> scale by timestep
    
    def solve_ode(self, x0, t, x):
        self.x = x  # overwrites
        self.dt = t
        outputs = odeint(self, x0, torch.tensor([0,1.0]).to(self.device),rtol=1e-3, atol=1e-3)[1]
        return outputs
    
class NeuralODE(Baseline):

    def __init__(self, input_dim, hidden_dim, p, output_dim, batch_size,device):
        Baseline.__init__(self,input_dim, hidden_dim, p, output_dim, device)
        self.device = device
        self.func = NeuralODE_(input_dim,batch_size,device).to(device)
        
    def forward(self, dt, x, p=0.0):
        
        #x = x.squeeze(0)
        #dt = dt.squeeze(0)
        T = x.size(1)
        
        # ODE
        mu_out = torch.zeros(x.size(0),T,1,device = self.device)
        for i in range(0,T):
            y0 = x[:,i:(i+1),0:1]
            x_i = x[:,i:(i+1),1:]
            dt_i = (dt[:,i,:][:,1] - dt[:,i,:][:,0]).unsqueeze(1).unsqueeze(1)
            mu_out[:,i:(i+1),:] = self.func.solve_ode(y0,dt_i,x_i)

        return mu_out
    
    def loss_fn(self,y_,y,msk):
        return torch.mean((y_[msk] - y[msk])**2)

#-------------------------------------------------------------------------------------------------

class latentODE2_(nn.Module):
    """
    dglucose/dt = NN(glucose,insulin)
    """
    def __init__(self,hidden_dim,input_dim,batch_size,device):
        super(latentODE2_, self).__init__()
        
        self.x = torch.zeros(batch_size,SEQUENCE_LENGTH,input_dim).to(device)
        self.dt = torch.zeros(batch_size,SEQUENCE_LENGTH,1).to(device)
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(input_dim+hidden_dim, 20),
            nn.ReLU(),
            nn.Linear(20, hidden_dim),
            nn.Tanh(),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, z):
        xz = torch.cat((z,self.x),2)
        return self.net(xz)*(self.dt*DT_SCALER) # -> scale by timestep
    
    def solve_ode(self, z0, t, x):
        self.x = x  # overwrites
        self.dt = t
        outputs = odeint(self, z0, torch.tensor([0,1.0]).to(self.device),rtol=1e-3, atol=1e-3)[1]
        return outputs
    
class LatentODE1(Baseline):

    def __init__(self, input_dim, hidden_dim, p, output_dim, batch_size,device):
        Baseline.__init__(self,input_dim, hidden_dim, p, output_dim, device)
        self.device = device
        self.func = latentODE2_(hidden_dim,input_dim,batch_size,device).to(device)
        self.mu_net = nn.Sequential(
            nn.Linear(hidden_dim, 10),
            nn.Tanh(),
            nn.Linear(10, 1)
        ).to(device)
        self.encode_z0 = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.Tanh(),
            nn.Linear(10, hidden_dim),
            nn.Tanh(),
        ).to(device)
        
        for m in self.encode_z0.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
        
    def forward(self, dt, x, p=0.0):
        
        #x = x.squeeze(0)
        batch_size = x.size(0)
        T = x.size(1)
        
        # ODE
        mu_out = torch.zeros(batch_size,T,1,device = self.device)
        #z0 = 0.01 * torch.randn(batch_size,1,self.hidden_dim,device = self.device)
        z0 = self.encode_z0(x[:,0,:]).unsqueeze(1)
        for i in range(0,T):
            x_i = x[:,i:(i+1),:]
            dt_i = (dt[:,i,:][:,1] - dt[:,i,:][:,0]).unsqueeze(1).unsqueeze(1)
            z0 = self.func.solve_ode(z0,dt_i,x_i)
            mu_out[:,i:(i+1),:] = self.mu_net(z0.squeeze(1)).unsqueeze(1)

        return mu_out
    
    def loss_fn(self,y_,y,msk):
        return torch.mean((y_[msk] - y[msk])**2)
        
#-------------------------------------------------------------------------------------------------

class ODERNN_(nn.Module):
    """
    In an ODE-RNN the hidden state h_t of the RNN evolves according to
    an ODE. This ODE is a neural network, i.e. dh/dt = ODEFunc(h,x).
    """
    def __init__(self,input_dim,hidden_dim,batch_size,device):
        super(ODERNN_, self).__init__()

        self.x = torch.zeros(batch_size,SEQUENCE_LENGTH,input_dim).to(device)
        self.dt = torch.zeros(batch_size,SEQUENCE_LENGTH,1).to(device)
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 50),
            nn.Tanh(),
            nn.Linear(50, hidden_dim),
            nn.Tanh(),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, dt, y):
        return self.net(y)*(self.dt*DT_SCALER)
    
    def solve_ode(self, z0, t, x):
        self.x = x  # overwrites
        self.dt = t
        outputs = odeint(self, z0, torch.tensor([0,1.0]).to(self.device),rtol=1e-1, atol=1e-1)[1]
        return outputs

class ODERNN(Baseline):
    """
    ODE-RNN
    """
    def __init__(self, input_dim, hidden_dim, p, output_dim, batch_size,device):
        Baseline.__init__(self,input_dim, hidden_dim, p, output_dim, device)
        # ODE-RNN
        self.rnn = nn.RNNCell(input_dim, hidden_dim)
        self.func = ODERNN_(input_dim,hidden_dim,batch_size,device)
        # N(mu,sigma)
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(hidden_dim,hidden_dim//2)
        self.distribution_mu = nn.Linear(hidden_dim//2, 1)

    def forward(self, dt, x, p=0.0):
        
        T = x.size(1)

        batch_size = x.size(0)
        mu_out = torch.zeros(batch_size,T,1,device = self.device)
        h_t = torch.zeros(batch_size, self.rnn.hidden_size,device=self.device)
        for i in range(0,T):
            x_i = x[:,i:(i+1),:]
            dt_i = (dt[:,i,:][:,1] - dt[:,i,:][:,0]).unsqueeze(1)
            h_t = self.rnn(x_i.squeeze(1),h_t)
            h_t = self.func.solve_ode(h_t,dt_i,x_i)
            mu = self.l1(h_t)
            mu = F.dropout(mu,training=True,p=p)
            mu = self.relu(mu)
            mu_out[:,i:(i+1),:] = self.distribution_mu(mu).unsqueeze(1)

        return mu_out
    
    def loss_fn(self,y_,y,msk):
        return torch.mean((y_[msk] - y[msk])**2)

#-------------------------------------------------------------------------------------------------

class LatentODE2_(nn.Module):
    """
    (dglucose/dt,dh/dt) = (RNN(glucose,h),NN(h,x))
    """
    def __init__(self,hidden_dim,input_dim,batch_size,device):
        super(LatentODE2_, self).__init__()

        self.x = torch.zeros(batch_size,SEQUENCE_LENGTH,input_dim).to(device)
        self.dt = torch.zeros(batch_size,SEQUENCE_LENGTH,1).to(device)
        self.device = device
        # dh/dt
        self.dh = nn.Sequential(
            nn.Linear(hidden_dim+input_dim, (hidden_dim+input_dim)*2),
            nn.Tanh(),
            nn.Linear((hidden_dim+input_dim)*2, hidden_dim),
            nn.Tanh(),
        ).to(device)
        # dy/dt
#         self.dy_rnn = nn.RNNCell(input_dim, hidden_dim)
#         self.dy_lin = nn.Linear(hidden_dim, 1)
#         self.dy_tanh = nn.Tanh()
        self.dy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.Tanh(),
            nn.Linear(hidden_dim*2, 1)
        ).to(device)

        for m in self.dh.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
        nn.init.normal_(self.dy_rnn.weight_ih,mean=0, std=0.1)
        nn.init.normal_(self.dy_rnn.weight_hh,mean=0, std=0.1)
        nn.init.constant_(self.dy_rnn.bias_ih, val=0)
        nn.init.constant_(self.dy_rnn.bias_ih, val=0)
        nn.init.normal_(self.dy_lin.weight,mean=0, std=0.1)
        nn.init.constant_(self.dy_lin.bias, val=0)

    def forward(self, t, z):
        # dh/dt
        #dh = self.dh(z[:,:,1:])*self.dt
        dh = self.dh(torch.cat((self.x,z[:,:,1:]),2)) * (self.dt * DT_SCALER)
        # dy/dt
        #dy = self.dy_rnn(self.x.squeeze(1),z[:,:,1:].squeeze(1)) # hidden should be shape (1,batch,hidden)
        #dy = self.dy_lin(dy.unsqueeze(1))
        dy = self.dy(z[:,:,1:]) * (self.dt * DT_SCALER)
        return torch.cat((dy,dh),2)

    def solve_ode(self, z0, t, x):
        self.x = x  # overwrites
        self.dt = t
        outputs = odeint(self, z0, torch.tensor([0,1.0]).to(self.device),rtol=1e-1, atol=1e-1)[1]
        return outputs    
    
class LatentODE2(Baseline):

    def __init__(self, input_dim, hidden_dim, p, output_dim,batch_size, device):
        Baseline.__init__(self,input_dim, hidden_dim, p, output_dim, device)
        self.device = device
        self.batch_size = batch_size
        self.func = LatentODE2_(hidden_dim,input_dim,batch_size,device).to(device)
        
    def forward(self, dt, x, seqlen=0):
        
        #x = x.squeeze(0)
        batch_size = x.size(0)
        T = x.size(1)

        # ODE stuff
        mu_out = torch.zeros(batch_size,T,1,device = self.device)
        z0 = torch.zeros(batch_size,1,self.hidden_dim+1,device = self.device)
        for i in range(0,T):
            x_i = x[:,i:(i+1),:]
            y0 = x[:,i:(i+1),0:1]
            z0[:,:,0:1] = y0
            dt_i = (dt[:,i,:][:,1] - dt[:,i,:][:,0]).unsqueeze(1).unsqueeze(1)
            z0 = self.func.solve_ode(z0,dt_i,x_i)
            mu_out[:,i:(i+1),:] = z0[:,:,0:1]

        return mu_out
    
    def loss_fn(self,y_,y,msk):
        return torch.mean((y_[msk] - y[msk])**2)