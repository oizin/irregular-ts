#!/bin/sh
python run_training.py --net NeuralODE --nepochs=10 --hypertune 1
python run_training.py --net ODERNN --nepochs=10 --hypertune 1
python run_training.py --net LatentODE1 --nepochs=10 --hypertune 1
python run_training.py --net LatentODE2 --nepochs=10 --hypertune 1