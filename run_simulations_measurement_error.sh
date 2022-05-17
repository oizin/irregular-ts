#!/bin/sh
echo "0.5 error\n"
for s in 1 2 3
do
echo "0.5 error $s\n"
julia scripts/simulations/simulate_glucose.jl 10000 0.5 $s
python simulation.py --net ctRNNModel --max_epochs 100 --logfolder measurement_error --N 10000 --sim_error 0.5
python simulation.py --net ctGRUModel --max_epochs 100 --logfolder measurement_error --N 10000 --sim_error 0.5
python simulation.py --net ctLSTMModel --max_epochs 100 --logfolder measurement_error --N 10000 --sim_error 0.5
python simulation.py --net ODEGRUBayes --max_epochs 100 --logfolder measurement_error --N 10000 --sim_error 0.5
python simulation.py --net ODEGRUBayes --max_epochs 100 --logfolder measurement_error --N 10000 --loss KL --sim_error 0.5 --merror 0.2
python simulation.py --net neuralJumpModel --max_epochs 100 --logfolder measurement_error --N 10000 --sim_error 0.5
python simulation.py --net resNeuralJumpModel --max_epochs 100 --logfolder measurement_error --N 10000 --sim_error 0.5
python simulation.py --net IMODE --max_epochs 100 --logfolder measurement_error --N 10000 --sim_error 0.5
done

echo "1.0 error\n"
for s in 1 2 3
do
echo "1.0 error $s\n"
julia scripts/simulations/simulate_glucose.jl 10000 1.0 $s
python simulation.py --net ctRNNModel --max_epochs 100 --logfolder measurement_error --N 10000 --sim_error 1.0
python simulation.py --net ctGRUModel --max_epochs 100 --logfolder measurement_error --N 10000 --sim_error 1.0
python simulation.py --net ctLSTMModel --max_epochs 100 --logfolder measurement_error --N 10000 --sim_error 1.0
python simulation.py --net ODEGRUBayes --max_epochs 100 --logfolder measurement_error --N 10000 --sim_error 1.0
python simulation.py --net ODEGRUBayes --max_epochs 100 --logfolder measurement_error --N 10000 --loss KL --sim_error 1.0 --merror 0.3
python simulation.py --net neuralJumpModel --max_epochs 100 --logfolder measurement_error --N 10000 --sim_error 1.0
python simulation.py --net resNeuralJumpModel --max_epochs 100 --logfolder measurement_error --N 10000 --sim_error 1.0
python simulation.py --net IMODE --max_epochs 100 --logfolder measurement_error --N 10000 --sim_error 1.0
done