# use lightning framework
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from argparse import ArgumentParser
# data related
from src.data.data_loader import MIMIC3DataModule
from data.feature_sets import all_features,glycaemic_features
# import models
from src.models.base import BaseModel
from src.models.models import *

# possible models
nets = {'ctRNNModel': ctRNNModel,
        'ctGRUModel': ctGRUModel,
        'ctLSTMModel':ctLSTMModel,
        'latentJumpModel':latentJumpModel,
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
parser = BaseModel.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()
dict_args = vars(args)

# seed
seed_everything(dict_args['seed'], workers=True)

# data
input_features = all_features()
mimic3 = MIMIC3DataModule(input_features,batch_size=128)

# model
input_dim = len(input_features)
net = nets[dict_args['net']]
model = net(input_dim,dict_args['hidden_dim'],dict_args['lr'])

# logging
logger = CSVLogger("experiments/mimic3",name=dict_args['logfolder'])
lr_monitor = LearningRateMonitor(logging_interval='step')

# train
early_stopping = EarlyStopping(monitor="val_loss",mode="min",verbose=True,patience=3,min_delta=0.0)  # mostly defaults
trainer = pl.Trainer.from_argparse_args(args,
                    logger=logger,
                    val_check_interval=0.5,
                    log_every_n_steps=50,
                    callbacks=[lr_monitor,early_stopping])
trainer.fit(model, mimic3)

# test
if dict_args['test'] == True:
    trainer.test(model,mimic3,ckpt_path="best")