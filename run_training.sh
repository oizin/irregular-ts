#!/bin/sh
python run_experiments.py --net LatentODE1 --nepochs=20 --hypertune 1 --gpu 1 --batchsize=128
python run_experiments.py --net ODERNN --nepochs=20 --hypertune 1 --gpu 1 --batchsize=128
python run_experiments.py --net ODEGRU --nepochs=20 --hypertune 1 --gpu 1 --batchsize=128
python run_experiments.py --net ODELSTM --nepochs=20 --hypertune 1 --gpu 1 --batchsize=128
python run_experiments.py --net LatentODE1 --nepochs=20 --hypertune 0 --gpu 1 --batchsize=128
python run_experiments.py --net ODERNN --nepochs=20 --hypertune 0 --gpu 1 --batchsize=128
python run_experiments.py --net ODEGRU --nepochs=20 --hypertune 0 --gpu 1 --batchsize=128
python run_experiments.py --net ODELSTM --nepochs=20 --hypertune 0 --gpu 1 --batchsize=128


python run_training.py --net ODEGRU --nepochs=20 --hypertune 1 --gpu 1 --batchsize=128
