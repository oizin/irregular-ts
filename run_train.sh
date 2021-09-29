#!/bin/sh
python train.py --hidden_dim=12 --net=ctRNNModel --gpus=0 --max_epochs=$1 --logfolder=$2 --test=$3
python train.py --hidden_dim=12 --net=ctGRUModel --gpus=0 --max_epochs=$1 --logfolder=$2 --test=$3
python train.py --hidden_dim=12 --net=ctLSTMModel --gpus=0 --max_epochs=$1 --logfolder=$2 --test=$3
python train.py --hidden_dim=30 --net=latentJumpModel --gpus=1 --max_epochs=$1 --logfolder=$2 --test=$3
python train.py --hidden_dim=12 --net=dtRNNModel --gpus=1 --max_epochs=$1 --logfolder=$2 --test=$3
python train.py --hidden_dim=12 --net=dtGRUModel --gpus=1 --max_epochs=$1 --logfolder=$2 --test=$3
python train.py --hidden_dim=12 --net=dtLSTMModel --gpus=1 --max_epochs=$1 --logfolder=$2 --test=$3