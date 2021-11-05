#!/bin/sh
python simulation.py --net ctRNNModel --max_epochs 20 --logfolder sanity --N 5000
python simulation.py --net ctGRUModel --max_epochs 20 --logfolder sanity --N 5000
python simulation.py --net ODEGRUBayes --max_epochs 20 --logfolder sanity --N 5000
python simulation.py --net ctLSTMModel --max_epochs 20 --logfolder sanity --N 5000
python simulation.py --net neuralJumpModel --max_epochs 20 --logfolder sanity --N 5000
python simulation.py --net resNeuralJumpModel --max_epochs 20 --logfolder sanity --N 5000
python simulation.py --net dtRNNModel --max_epochs 20 --logfolder sanity --N 5000
python simulation.py --net dtGRUModel --max_epochs 20 --logfolder sanity --N 5000
python simulation.py --net dtLSTMModel --max_epochs 20 --logfolder sanity --N 5000
