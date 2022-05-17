#!/bin/sh
echo "0.5 error\n"
for s in 1 2 3
do
echo "0.5 error $s\n"
julia scripts/simulations/simulate_glucose.jl 20000 1.0 $s
python simulation.py --net ctRNNModel --max_epochs 100 --logfolder measurement_error --N 20000 --sim_error 1.0
python simulation.py --net ctGRUModel --max_epochs 100 --logfolder measurement_error --N 20000 --sim_error 1.0
python simulation.py --net ctLSTMModel --max_epochs 100 --logfolder measurement_error --N 20000 --sim_error 1.0
python simulation.py --net ODEGRUBayes --max_epochs 100 --logfolder measurement_error --N 20000 --sim_error 1.0
python simulation.py --net ODEGRUBayes --max_epochs 100 --logfolder measurement_error --N 20000 --loss KL --sim_error 1.0 --merror 0.05
python simulation.py --net neuralJumpModel --max_epochs 100 --logfolder measurement_error --N 20000 --sim_error 1.0
python simulation.py --net resNeuralJumpModel --max_epochs 100 --logfolder measurement_error --N 20000 --sim_error 1.0
python simulation.py --net IMODE --max_epochs 100 --logfolder measurement_error --N 20000 --sim_error 1.0
done

echo "1.0 error\n"
for s in 1 2 3
do
echo "1.0 error $s\n"
julia scripts/simulations/simulate_glucose.jl 20000 2.0 $s
python simulation.py --net ctRNNModel --max_epochs 100 --logfolder measurement_error --N 20000 --sim_error 2.0
python simulation.py --net ctGRUModel --max_epochs 100 --logfolder measurement_error --N 20000 --sim_error 2.0 
python simulation.py --net ctLSTMModel --max_epochs 100 --logfolder measurement_error --N 20000 --sim_error 2.0
python simulation.py --net ODEGRUBayes --max_epochs 100 --logfolder measurement_error --N 20000 --sim_error 2.0
python simulation.py --net ODEGRUBayes --max_epochs 100 --logfolder measurement_error --N 20000 --loss KL --sim_error 2.0 --merror 0.05
python simulation.py --net neuralJumpModel --max_epochs 100 --logfolder measurement_error --N 20000 --sim_error 2.0
python simulation.py --net resNeuralJumpModel --max_epochs 100 --logfolder measurement_error --N 20000 --sim_error 2.0
python simulation.py --net IMODE --max_epochs 100 --logfolder measurement_error --N 20000 --sim_error 2.0
done