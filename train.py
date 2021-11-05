# use lightning framework
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
# data related
from src.data.data_loader import MIMIC3DataModule
from data.feature_sets import all_features_treat_dict
# import models
from src.models.base import BaseModel
from src.models.models import *

# possible models
nets = {'ctRNNModel': ctRNNModel,
        'ctGRUModel': ctGRUModel,
        'ctLSTMModel':ctLSTMModel,
        'neuralJumpModel':neuralJumpModel,
        'resNeuralJumpModel':resNeuralJumpModel,
        'ODEGRUBayes':ODEGRUBayes,
        'dtRNNModel':dtRNNModel,
        'dtGRUModel':dtGRUModel,
        'dtLSTMModel':dtLSTMModel}

# cmd args
parser = ArgumentParser()
parser.add_argument('--net', dest='net',choices=list(nets.keys()),type=str)
parser.add_argument('--seed', dest='seed',default=42,type=int)
parser.add_argument('--lr', dest='lr',default=0.01,type=float)
parser.add_argument('--test', dest='test',default=False,type=bool)
parser.add_argument('--logfolder', dest='logfolder',default='default',type=str)
parser.add_argument('--update_loss', dest='update_loss',default=0.1,type=float)
parser = BaseModel.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()
dict_args = vars(args)

# seed
seed_everything(dict_args['seed'], workers=True)

# data
input_features = all_features_treat_dict()
mimic3 = MIMIC3DataModule(input_features,batch_size=128,testing = False)

# model
input_dims = {'input_dim_t':len(input_features['timevarying']),'input_dim_0':len(input_features['static']),'input_dim_i':len(input_features['intervention'])}
hidden_dims = {'hidden_dim_t':dict_args['hidden_dim_t'],'hidden_dim_0':dict_args['hidden_dim_0']}
print(input_dims)
print(hidden_dims)
preNN = nn.Sequential(
    nn.Linear(input_dims['input_dim_t']+hidden_dims['hidden_dim_0'],max((input_dims['input_dim_t']+hidden_dims['hidden_dim_0'])//2,1)),
    nn.Tanh(),
    nn.Dropout(p=0.2),
    nn.Linear(max((input_dims['input_dim_t']+hidden_dims['hidden_dim_0'])//2,1),hidden_dims['hidden_dim_t']),
    nn.Tanh(),
)
NN0 = nn.Sequential(
    nn.Linear(input_dims['input_dim_0'],max(input_dims['input_dim_0']//2,1)),
    nn.Tanh(),
    nn.Dropout(p=0.2),
    nn.Linear(max(input_dims['input_dim_0']//2,1), hidden_dims['hidden_dim_0']),
    nn.Tanh(),
)
net = nets[dict_args['net']]
model = net(input_dims,hidden_dims,preNN,NN0,dict_args['lr'],dict_args['update_loss'])

# logging
logger = CSVLogger("experiments/mimic3",name=dict_args['logfolder'])
lr_monitor = LearningRateMonitor(logging_interval='step')
checkpoint_callback = ModelCheckpoint(monitor='val_loss',save_top_k=1)

# train
early_stopping = EarlyStopping(monitor="val_loss",mode="min",verbose=True,patience=5,min_delta=0.0)  # mostly defaults
trainer = pl.Trainer.from_argparse_args(args,
                    logger=logger,
                    val_check_interval=1.0,
                    log_every_n_steps=50,
                    callbacks=[lr_monitor,early_stopping,checkpoint_callback])
trainer.fit(model, mimic3)

# test
if dict_args['test'] == True:
    trainer.test(model,mimic3,ckpt_path="best")