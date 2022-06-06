
To do:

* Rename train.py to mimic.py
* mimic.py should do cross validation: folds, seed, save_predictions as arguments
* simulated data should be created and saved in data: simulation_dataset.ipynb
* put all experiments in experiments.sh
* there should be poetry dependency set up

Performance:  
* rerun

Code for the following:

<graph>

Demo run

A demo version of the data used, from the openly available, can be found in the folder XXX. This
enables the user to see the final data structure and run the code in the . See below for more 
details on repeating the full analysis. 

### Data setup

While we cannot publicaly share the full data used, it is available to those with
credentialed access to physionet.org. Credentialed access can be requested at XXX and requires
completion of XXX and an second academic reference who can xxx. 

A demo version of the data used, from the openly available, can be found in the folder XXX. This
enables the user to see the final data structure and run the code in the 

See dbt project at github.com/mimic

The final output of that project is downloaded via analysis_dataset.ipynb

### Installing torchctrnn
    
Install from Github as follows:
        
pip install https://github.com/oizin/torchctrnn/tarball/main

NOTE: add an irregular-ts branch to torchctrnn!
    
### Using the code
    
The files mimic.py and simulation.py
    
python train.py --hidden_dim_t=12 --net=ctRNNModel --max_epochs=100 --update_loss=0.1 --logfolder=model_results_19112021 --test=True --nfolds=1 --seed 1


