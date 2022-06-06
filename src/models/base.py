import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt 

class BaseModel(pl.LightningModule):
    """BaseModel
    
    loss function a property of output NN
    
    to do:
        - add in hidden, output dims (entering twice no issue!)
        
    for forward methods see...
    
    input_dims - a dictionary
    """
    def __init__(self,RNN,OutputNN,preNN,NN0,dims,ginv,eval_fn,learning_rate=1e-1,update_mixing=1e-3,merror=1e-2):
        super().__init__()
        #self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.RNN = RNN
        self.OutputNN = OutputNN
        self.loss_fn = OutputNN.loss_fn
        self.dims = dims
        self.hidden_dim_t = dims['hidden_dim_t']
        self.hidden_dim_0 = dims['hidden_dim_0']
        self.input_dim_t = dims['input_dim_t']
        self.input_dim_i = dims['input_dim_i']
        self.input_dim_0 = dims['input_dim_0']
        self.update_mixing = update_mixing
        self.preNN = preNN
        self.NN0 = NN0
        self.merror=merror
        self.ginv=ginv
        self.loss_update_fn = OutputNN.loss_update_fn
        self.eval_fn = eval_fn

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BaseModel")
        parser.add_argument("--hidden_dim_t", type=int, default=12)
        parser.add_argument("--hidden_dim_0", type=int, default=12)
        parser.add_argument("--hidden_dim_i", type=int, default=8)
        return parent_parser
            
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        optim_setting = {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=5,threshold=1e-2),
            "monitor": "val_loss"}}
        return optim_setting

    def training_step(self, batch, batch_idx):
        xt,x0,xi,y,msk,dt,msk_update,_ = batch
        msk = msk.bool()
        msk_update = msk_update.bool()
        pred_step,pred_update_step = self.forward(dt,(xt,x0,xi),training = True, p = 0.2,include_update=True)
        loss_pred = self.loss_fn(pred_step[~msk],y[~msk])
        loss_update = self.loss_update_fn(pred_update_step[~msk_update],xt[:,:,0][~msk_update],e=self.merror)
        loss_step = loss_pred + self.update_mixing*loss_update
        self.log("training_loss", loss_step)
        self.log("pred_loss", loss_pred)
        self.log("update_loss", loss_update)
        return loss_step
    
    def validation_step(self, batch, batch_idx):
        xt,x0,xi,y, msk, dt, msk_update,_ = batch
        msk = msk.bool()
        msk_update = msk_update.bool()
        pred_step,pred_update_step = self.forward(dt,(xt,x0,xi),include_update=True)
        loss_pred = self.loss_fn(pred_step[~msk],y[~msk])
        loss_update = self.loss_update_fn(pred_update_step[~msk_update],xt[:,:,0][~msk_update],e=self.merror)
        loss_step = loss_pred + self.update_mixing*loss_update
        n_step = torch.sum(~msk.bool())
        return {'loss':loss_step,'loss_update':loss_update,'loss_pred':loss_pred,'n':n_step}
    
    def validation_epoch_end(self, outputs):
        loss_epoch = torch.tensor([o['loss'] for o in outputs])
        loss_pred_epoch = torch.tensor([o['loss_pred'] for o in outputs])
        loss_update_epoch = torch.tensor([o['loss_update'] for o in outputs])
        n_epoch = torch.tensor([o['n'] for o in outputs])
        loss = torch.sum(loss_epoch)#/torch.sum(n_epoch)
        loss_pred = torch.sum(loss_pred_epoch)#/torch.sum(n_epoch)
        loss_update = torch.sum(loss_update_epoch)#/torch.sum(n_epoch)
        self.log("val_loss", loss)
        self.log("val_loss_pred", loss_pred)
        self.log("val_loss_update", loss_update)

    def test_step(self, batch, batch_idx):
        xt,x0,xi, y, msk, dt,_,_ = batch
        msk = msk.bool()
        pred_step = self.forward(dt,(xt,x0,xi))
        loss_step = self.loss_fn(pred_step[~msk],y[~msk])
        n_step = torch.sum(~msk.bool())
        return {'loss':loss_step,'n':n_step,'y':y[~msk].cpu(),'pred':pred_step[~msk].cpu()}

    def test_epoch_end(self, outputs):
        # extract data
        # loss_update_epoch = torch.tensor([o['loss_update'] for o in outputs])
        # loss_pred_epoch = torch.tensor([o['loss_pred'] for o in outputs])
        loss_epoch = torch.tensor([o['loss'] for o in outputs])
        n_epoch = torch.tensor([o['n'] for o in outputs])
        y_epoch = np.concatenate([o['y'] for o in outputs])
        y_epoch = np.expand_dims(y_epoch,1)
        pred_epoch = np.concatenate([o['pred'] for o in outputs])
        # pred_update = np.concatenate([o['pred_update'] for o in outputs])
        # metrics
        loss = torch.sum(loss_epoch)/torch.sum(n_epoch)
        # log
        self.log("test_loss", loss)
        eval = self.eval_fn(pred_epoch,y_epoch,ginv=self.ginv)
        print(eval)
        for key in eval.keys():
            self.log(key,eval[key])

    def forward(self, dt, x, training = False, p = 0.2, include_update=False):
        """
        x a tuple
        """
        xt,x0,xi = x
        T = xt.size(1)
        batch_size = xt.size(0)
        output = torch.zeros(batch_size,T,self.OutputNN.output_dim,device = self.device)
        output_update = torch.zeros(batch_size,T,self.OutputNN.output_dim,device = self.device)
        h_t = torch.zeros(batch_size, self.hidden_dim_t,device=self.device)
        z0 = self.NN0(x0)
        for i in range(0,T):
            xt_i = xt[:,i,:]
            xi_i = xi[:,i,:]
            xt_i = self.preNN(torch.cat((xt_i,z0),1))
            dt_i = dt[:,i,:]
            if (include_update == True):
                h_t_update = self.RNN.forward_update(xt_i,h_t)
                h_t = self.RNN.forward_ode(h_t_update,dt_i,xi_i)
                h_t_update = F.dropout(h_t_update,training=training,p=p)
                h_t = F.dropout(h_t,training=training,p=p)
                output_update[:,i,:] = self.OutputNN(h_t_update)
                output[:,i,:] = self.OutputNN(h_t)
            else:
                h_t = self.RNN(xt_i,h_t,dt_i,xi_i)
                output[:,i,:] = self.OutputNN(h_t)
        if (include_update == True):
            return output,output_update
        else:
            return output
    
    # broken!
    # def forward_trajectory(self, dt, x, nsteps=10):
    #     xt, x0,xi = x
    #     T = xt.size(1)
    #     batch_size = xt.size(0)
    #     outputs = []
    #     h_t = torch.zeros(batch_size, self.hidden_dim_t,device=self.device)
    #     if (self.NN0 != None):
    #         z0 = self.NN0(x0)
    #     for i in range(0,T):
    #         xt_i = xt[:,i,:]
    #         xi_i = xi[:,i,:]
    #         if (self.NN0 != None) & (self.preNN != None):
    #             xt_i = self.preNN(torch.cat((xt_i,z0),1))
    #         elif (self.preNN != None):
    #             xt_i = self.preNN(xt_i)
    #         dt_i = dt[:,i,:]
    #         h_t = self.RNN(xt_i,h_t,dt_i,xi_i,n_intermediate=nsteps).squeeze(0)
    #         outputs_i = self.OutputNN(h_t)
    #         outputs.append(outputs_i)
    #         h_t = h_t[-1]
    #     return outputs

    def predict_step(self,batch, batch_idx, dataloader_idx=0):
        xt,x0,xi,_,msk,dt,_,key = batch
        msk = msk.bool()
        output = self.forward(dt, (xt,x0,xi))
        output = output[~msk.bool().repeat_interleave(1,1)]
        key = key[~msk.bool()]
        key = key.repeat_interleave(1,0)
        return torch.cat((key.unsqueeze(1),output),1)


class BaseModelAblate(BaseModel):
    """BaseModelAblate
    
    no time
    """
    def __init__(self,RNN,OutputNN,preNN,NN0,dims,ginv,eval_fn,**kwargs):
        super().__init__(RNN,OutputNN,preNN,NN0,dims,ginv,eval_fn,**kwargs)

    def training_step(self, batch, batch_idx):
        xt,x0,xi,y, msk, dt, _,_ = batch
        msk = msk.bool()
        pred_step = self.forward(dt,(xt,x0,xi),training = True, p = 0.2)
        loss_step = self.loss_fn(pred_step[~msk],y[~msk])
        self.log("training_loss", loss_step)
        return loss_step
    
    def validation_step(self, batch, batch_idx):
        xt,x0,xi,y, msk, dt, _,_ = batch
        msk = msk.bool()
        pred_step = self.forward(dt,(xt,x0,xi))
        loss_step = self.loss_fn(pred_step[~msk],y[~msk])
        n_step = torch.sum(~msk.bool())
        return {'loss':loss_step,'n':n_step}
    
    def validation_epoch_end(self, outputs):
        loss_epoch = torch.tensor([o['loss'] for o in outputs])
        n_epoch = torch.tensor([o['n'] for o in outputs])
        loss = torch.sum(loss_epoch)/torch.sum(n_epoch)
        self.log("val_loss", loss)

    def forward(self, dt, x, training = False, p = 0.2):
        
        xt,x0,xi = x
        T = xt.size(1)
        batch_size = xt.size(0)
        output = torch.zeros(batch_size,T,self.OutputNN.output_dim,device = self.device)
        h_t = torch.zeros(batch_size, self.hidden_dim_t,device=self.device)
        if (self.NN0 != None):
            z0 = self.NN0(x0)
        for i in range(0,T):
            xt_i = xt[:,i,:]
            xi_i = xi[:,i,:]
            if (self.NN0 != None) & (self.preNN != None):
                xt_i = self.preNN(torch.cat((xt_i,z0),1))
            elif (self.preNN != None):
                xt_i = self.preNN(xt_i)
            dt_i = dt[:,i,:]
            h_t = self.RNN(xt_i,h_t).squeeze(0)
            h_t = F.dropout(h_t,training=training,p=p)
            output[:,i,:] = self.OutputNN(h_t)
        return output    


class BaseModelForward(BaseModel):
    """BaseModelForward
    
    no recurrence
    """
    def __init__(self,OutputNN,preNN,NN0,dims,ginv,eval_fn,**kwargs):
        super().__init__(None,OutputNN,preNN,NN0,dims,ginv,eval_fn,**kwargs)












# class BaseModelCT(BaseModel):
#     """BaseModelCT
    
#     continuous time
#     """
#     def __init__(self,RNN,OutputNN,preNN,NN0,dims,learning_rate=1e-2,update_loss=0.1,merror=1e-5):
#         super().__init__(RNN,OutputNN,preNN,NN0,dims,learning_rate,update_loss,merror)
#         self.loss_update_fn = OutputNN.loss_update_fn
        
#     def training_step(self, batch, batch_idx):
#         xt,x0,xi,y,msk,dt,msk_update,_ = batch
#         msk = msk.bool()
#         msk_update = msk_update.bool()
#         pred_step,pred_update_step = self.forward(dt,(xt,x0,xi),training = True, p = 0.2,include_update=True)
#         loss_pred = self.loss_fn(pred_step[~msk],y[~msk])
#         loss_update = self.loss_update_fn(pred_update_step[~msk_update],xt[:,:,0][~msk_update],e=self.merror)
#         loss_step = loss_pred + self.update_loss_scale*self.update_loss*loss_update
#         self.log("training_loss", loss_step)
#         self.log("pred_loss", loss_pred)
#         self.log("update_loss", loss_update)
#         return loss_step
    
#     def validation_step(self, batch, batch_idx):
#         xt,x0,xi,y, msk, dt, msk_update,_ = batch
#         msk = msk.bool()
#         msk_update = msk_update.bool()
#         pred_step,pred_update_step = self.forward(dt,(xt,x0,xi),include_update=True)
#         loss_pred = self.loss_fn(pred_step[~msk],y[~msk])
#         loss_update = self.loss_update_fn(pred_update_step[~msk_update],xt[:,:,0][~msk_update],e=self.merror)
#         loss_step = loss_pred + self.update_loss*loss_update
#         sse_step = self.sse_fn(pred_step[~msk],y[~msk])
#         n_step = torch.sum(~msk.bool())
#         return {'loss':loss_step,'loss_update':loss_update,'loss_pred':loss_pred,'sse':sse_step,'n':n_step}
    
#     def validation_epoch_end(self, outputs):
#         loss_epoch = torch.tensor([o['loss'] for o in outputs])
#         loss_pred_epoch = torch.tensor([o['loss_pred'] for o in outputs])
#         loss_update_epoch = torch.tensor([o['loss_update'] for o in outputs])
#         sse_epoch = torch.tensor([o['sse'] for o in outputs])
#         n_epoch = torch.tensor([o['n'] for o in outputs])
#         rmse = torch.sqrt(torch.sum(sse_epoch)/torch.sum(n_epoch))
#         loss = torch.sum(loss_epoch)#/torch.sum(n_epoch)
#         loss_pred = torch.sum(loss_pred_epoch)#/torch.sum(n_epoch)
#         loss_update = torch.sum(loss_update_epoch)#/torch.sum(n_epoch)
# #         if (loss_pred > loss_update) and ((self.update_loss_scale*self.update_loss*loss_update) / loss_pred) > 2:
# #             self.update_loss_scale = self.update_loss_scale*0.5
#         self.log("val_loss", loss)
#         self.log("val_loss_pred", loss_pred)
#         self.log("val_loss_update", loss_update)
#         self.log("val_rmse", rmse)
        
#     def test_step(self, batch, batch_idx):
#         xt,x0,xi,y, msk, dt,msk_update,_ = batch
#         msk = msk.bool()
#         msk_update = msk_update.bool()
#         pred_step,pred_update_step = self.forward(dt,(xt,x0,xi),include_update=True)
#         loss_pred = self.loss_fn(pred_step[~msk],y[~msk])
#         loss_update = self.loss_update_fn(pred_update_step[~msk_update],xt[:,:,0][~msk_update],e=self.merror)
#         loss_step = loss_pred + self.update_loss*loss_update
#         sse_step = self.sse_fn(pred_step[~msk],y[~msk])
#         n_step = torch.sum(~msk.bool())
#         return {'loss_pred':loss_pred,'loss_update':loss_update,'loss_step':loss_step,
#                 'sse':sse_step,'n':n_step,'y':y[~msk].cpu(),'pred':pred_step[~msk].cpu(),'pred_update':pred_update_step[~msk_update].cpu()}
            
#     def test_epoch_end(self, outputs):
#         # extract data
#         loss_update_epoch = torch.tensor([o['loss_update'] for o in outputs])
#         loss_pred_epoch = torch.tensor([o['loss_pred'] for o in outputs])
#         loss_epoch = torch.tensor([o['loss_step'] for o in outputs])
#         sse_epoch = torch.tensor([o['sse'] for o in outputs])
#         n_epoch = torch.tensor([o['n'] for o in outputs])
#         y_epoch = np.concatenate([o['y'] for o in outputs])
#         pred_epoch = np.concatenate([o['pred'] for o in outputs])
#         pred_update = np.concatenate([o['pred_update'] for o in outputs])
#         # metrics
#         rmse = torch.sqrt(torch.sum(sse_epoch)/torch.sum(n_epoch))
#         loss_pred = torch.sum(loss_pred_epoch)/torch.sum(n_epoch)
#         loss_update = torch.sum(loss_update_epoch)/torch.sum(n_epoch)
#         loss = torch.sum(loss_epoch)/torch.sum(n_epoch)
#         # log
#         self.log("test_loss", loss)
#         self.log("test_loss_pred", loss_pred)
#         self.log("test_loss_update", loss_update)
#         self.log("test_rmse", rmse)
#         if pred_epoch.shape[1] > 1:
#             prob_eval = self.OutputNN.probabilistic_eval_fn(pred_epoch,y_epoch)
#             crps = prob_eval['crps_mean']
#             ig =  prob_eval['ig_mean']
#             int_score = prob_eval['int_score_mean']
#             var_pit = prob_eval['var_pit']
#             int_coverage = prob_eval['int_coverage']
#             int_av_width = prob_eval['int_av_width']
#             int_med_width = prob_eval['int_med_width']
#             # log
#             self.log("test_var_pit", var_pit)
#             self.log("test_crps", crps)
#             self.log("test_ignorance", ig)
#             self.log("test_int_score", int_score)
#             self.log("int_coverage", int_coverage)
#             self.log("int_med_width", int_med_width)
#             self.log("int_av_width", int_av_width)

    # def forward(self, dt, x, training = False, p = 0.2, include_update=False):
    #     """
    #     x a tuple
    #     """
    #     xt,x0,xi = x
    #     T = xt.size(1)
    #     batch_size = xt.size(0)
    #     output = torch.zeros(batch_size,T,self.OutputNN.output_dim,device = self.device)
    #     output_update = torch.zeros(batch_size,T,self.OutputNN.output_dim,device = self.device)
    #     h_t = torch.zeros(batch_size, self.hidden_dim_t,device=self.device)
    #     z0 = self.NN0(x0)
    #     for i in range(0,T):
    #         xt_i = xt[:,i,:]
    #         xi_i = xi[:,i,:]
    #         xt_i = self.preNN(torch.cat((xt_i,z0),1))
    #         dt_i = dt[:,i,:]
    #         if (include_update == True):
    #             h_t_update = self.RNN.forward_update(xt_i,h_t)
    #             h_t = self.RNN.forward_ode(h_t_update,dt_i,xi_i).squeeze(0)
    #             h_t_update = F.dropout(h_t_update,training=training,p=p)
    #             h_t = F.dropout(h_t,training=training,p=p)
    #             output_update[:,i,:] = self.OutputNN(h_t_update)
    #             output[:,i,:] = self.OutputNN(h_t)
    #         else:
    #             h_t = self.RNN(xt_i,h_t,dt_i,xi_i).squeeze(0)
    #             output[:,i,:] = self.OutputNN(h_t)
    #     if (include_update == True):
    #         return output,output_update
    #     else:
    #         return output
        
    # def forward_trajectory(self, dt, x, nsteps=10):
    #     xt, x0,xi = x
    #     T = xt.size(1)
    #     batch_size = xt.size(0)
    #     outputs = []
    #     h_t = torch.zeros(batch_size, self.hidden_dim_t,device=self.device)
    #     if (self.NN0 != None):
    #         z0 = self.NN0(x0)
    #     for i in range(0,T):
    #         xt_i = xt[:,i,:]
    #         xi_i = xi[:,i,:]
    #         if (self.NN0 != None) & (self.preNN != None):
    #             xt_i = self.preNN(torch.cat((xt_i,z0),1))
    #         elif (self.preNN != None):
    #             xt_i = self.preNN(xt_i)
    #         dt_i = dt[:,i,:]
    #         h_t = self.RNN(xt_i,h_t,dt_i,xi_i,n_intermediate=nsteps).squeeze(0)
    #         outputs_i = self.OutputNN(h_t)
    #         outputs.append(outputs_i)
    #         h_t = h_t[-1]
    #     return outputs

    # def predict_step(self,batch, batch_idx, dataloader_idx=0):
    #     xt,x0,xi,_,msk,dt,_,key = batch
    #     msk = msk.bool()
    #     output = self.forward(dt, (xt,x0,xi))
    #     output = output[~msk.bool().repeat_interleave(1,1)]
    #     key = key[~msk.bool()]
    #     key = key.repeat_interleave(1,0)
    #     return torch.cat((key.unsqueeze(1),output),1)

