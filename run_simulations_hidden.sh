#!/bin/sh
echo "10 hidden simulation\n"
julia scripts/data/simulate_ou.jl 10000 123
python simulation.py --net ctGRUModel --max_epochs 100 --logfolder hidden_dim --N 10000 --hidden_dim_t 10
julia scripts/data/simulate_ou.jl 20000 123
python simulation.py --net ctGRUModel --max_epochs 100 --logfolder hidden_dim --N 20000 --hidden_dim_t 10

echo "20 hidden simulation\n"
julia scripts/data/simulate_ou.jl 10000 123
python simulation.py --net ctGRUModel --max_epochs 100 --logfolder hidden_dim --N 10000 --hidden_dim_t 20
julia scripts/data/simulate_ou.jl 20000 123
python simulation.py --net ctGRUModel --max_epochs 100 --logfolder hidden_dim --N 20000 --hidden_dim_t 20

echo "50 hidden simulation\n"
julia scripts/data/simulate_ou.jl 10000 123
python simulation.py --net ctGRUModel --max_epochs 100 --logfolder hidden_dim --N 10000 --hidden_dim_t 50
julia scripts/data/simulate_ou.jl 20000 123
python simulation.py --net ctGRUModel --max_epochs 100 --logfolder hidden_dim --N 20000 --hidden_dim_t 50

echo "100 hidden simulation\n"
julia scripts/data/simulate_ou.jl 10000 123
python simulation.py --net ctGRUModel --max_epochs 100 --logfolder hidden_dim --N 10000 --hidden_dim_t 100
julia scripts/data/simulate_ou.jl 20000 123
python simulation.py --net ctGRUModel --max_epochs 100 --logfolder hidden_dim --N 20000 --hidden_dim_t 100
