#!/bin/sh
python train.py --hidden_dim_t=12 --net=ctRNNModel --gpus=0 --max_epochs=1 --update_loss=0.1 --logfolder=sanity --test=True
python train.py --hidden_dim_t=12 --net=ctGRUModel --gpus=1 --max_epochs=1 --update_loss=0.1 --logfolder=sanity --test=True
python train.py --hidden_dim_t=12 --net=ctLSTMModel --gpus=1 --max_epochs=1 --update_loss=0.1 --logfolder=sanity --test=True
python train.py --hidden_dim_t=30 --net=neuralJumpModel --gpus=1 --max_epochs=1 --update_loss=0.1 --logfolder=sanity --test=True
python train.py --hidden_dim_t=30 --net=resNeuralJumpModel --gpus=1 --max_epochs=1 --update_loss=0.1 --logfolder=sanity --test=True
python train.py --hidden_dim_t=30 --net=ODEGRUBayes --gpus=1 --max_epochs=1 --update_loss=0.1 --logfolder=sanity --test=True
python train.py --hidden_dim_t=12 --net=dtRNNModel --gpus=1 --max_epochs=1 --update_loss=0.1 --logfolder=sanity --test=True
python train.py --hidden_dim_t=12 --net=dtGRUModel --gpus=1 --max_epochs=1 --update_loss=0.1 --logfolder=sanity --test=True
python train.py --hidden_dim_t=12 --net=dtLSTMModel --gpus=1 --max_epochs=1 --update_loss=0.1 --logfolder=sanity --test=True