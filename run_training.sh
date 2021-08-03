#!/bin/sh
python run_training.py --net LatentODE1 --nepochs=50 --hypertune 0 --gpu 1
python run_training.py --net ODERNN --nepochs=30 --hypertune 0 --gpu 1
python run_training.py --net ODEGRU --nepochs=30 --hypertune 0 --gpu 1
python run_training.py --net ODELSTM --nepochs=30 --hypertune 0 --gpu 1