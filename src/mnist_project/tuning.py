"""Optuna studies use only the shared train/validation partition."""
from pathlib import Path
import tensorflow as tf
from .models import build_model, DEEP_MODELS
from .data import model_input
from .experiments import batches, write_json


def tune(name, data, output, trials=5, epochs=10):
    import optuna
    if name not in DEEP_MODELS or trials<1 or epochs<1:
        raise ValueError('Tune a deep model with positive trials and epochs.')
    directory = Path(output)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f'{directory} already contains files.')
    directory.mkdir(parents=True,exist_ok=True)
    x_train,x_val = model_input(data.x_train,name),model_input(data.x_val,name)

    def objective(trial):
        tf.keras.backend.clear_session()
        params={'learning_rate':trial.suggest_float('learning_rate',1e-4,1e-2,log=True),
                'dropout_rate':trial.suggest_float('dropout_rate',.2,.5)}
        if name!='tinyvit': params['num_filters']=trial.suggest_categorical('num_filters',[32,64])
        batch_size=trial.suggest_categorical('batch_size',[32,64,128])
        model=build_model(name,seed=data.seed,**params)
        history=model.fit(batches(x_train,data.y_train,batch_size,True,data.seed),
                          validation_data=batches(x_val,data.y_val,batch_size),epochs=epochs,verbose=0)
        return max(history.history['val_accuracy'])

    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=data.seed))
    study.optimize(objective,n_trials=trials)
    study.trials_dataframe().to_csv(directory/'trials.csv',index=False)
    result={'model':name,'best_validation_accuracy':study.best_value,'best_params':study.best_params,
            'seed':data.seed,'n_train':len(data.y_train),'n_val':len(data.y_val),'epochs_per_trial':epochs,
            'validation_size':data.validation_size,'data_source':data.source}
    write_json(directory/'best_params.json',result)
    return result
