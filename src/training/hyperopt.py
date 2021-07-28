def define_model(trial):

    hidden_dim = trial.suggest_int("hidden_dim", 6, 12)

    mean_fn = NET(FEATURE_DIM, hidden_dim, OUTPUT_DIM,DEVICE)
    model = NORE(mean_fn=mean_fn)

    return model

def objective(trial):
    
    # model 
    model = define_model(trial).to(DEVICE)
    
    # optimisation
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
    l2_penalty = trial.suggest_float("l2_penalty", 1e-3, 1e-1, log=True)    
    model_optim = getattr(optim, optimizer_name)(model.mean_fn.parameters(), lr=lr)
    
    # fixed
    loss_fn = nn.MSELoss()
    optim_scheduler = optim.lr_scheduler.ReduceLROnPlateau(model_optim, 'min')

    # train and evaluate
    dl_train = dataloaders['train']
    dl_val = dataloaders['validation']

    for epoch in range(N_EPOCHS):
        loss = train_single_epoch(model, dl_train, model_optim,loss_fn)
        error_val,y_preds, y_tests = evaluate(model, dl_val, loss_fn, 'validation')
        optim_scheduler.step(error_val)
            
        trial.report(error_val, epoch)
    
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return(error_val)