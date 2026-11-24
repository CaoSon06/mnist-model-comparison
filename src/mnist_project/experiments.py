"""Training, persistence, evaluation and inference shared by notebooks and CLI."""
import json
from pathlib import Path
from time import perf_counter
import platform
import importlib.metadata
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from .data import model_input
from .models import build_model, save_model, load_model, DEEP_MODELS


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def batches(x, y, batch_size, training=False, seed=42):
    import tensorflow as tf
    data = tf.data.Dataset.from_tensor_slices((x,y))
    if training:
        data = data.shuffle(min(len(y),10000),seed=seed,reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    return data.batch(batch_size).with_options(options).prefetch(1)


def raw_predictions(model, name, images, batch_size=256, include_scores=True):
    x = model_input(images,name)
    if name not in DEEP_MODELS:
        result = {'predictions':model.predict(x)}
        if include_scores and hasattr(model,'predict_proba'):
            result['probabilities'] = model.predict_proba(x)
            result['classes'] = model.classes_
        elif include_scores and hasattr(model,'decision_function'):
            result['decision_scores'] = model.decision_function(x)
            result['classes'] = model.classes_
        return result
    probs = np.concatenate([model(x[i:i+batch_size],training=False).numpy()
                            for i in range(0,len(x),batch_size)])
    return {'predictions':probs.argmax(axis=1),'probabilities':probs,'classes':np.arange(10)}


def metrics_for(labels, predictions):
    return {'n_test':len(labels), 'accuracy':float(accuracy_score(labels,predictions)),
            'precision_macro':float(precision_score(labels,predictions,average='macro',zero_division=0)),
            'recall_macro':float(recall_score(labels,predictions,average='macro',zero_division=0)),
            'f1_macro':float(f1_score(labels,predictions,average='macro',zero_division=0)),
            'f1_weighted':float(f1_score(labels,predictions,average='weighted',zero_division=0)),
            'classification_report':classification_report(labels,predictions,labels=list(range(10)),output_dict=True,zero_division=0),
            'confusion_matrix':confusion_matrix(labels,predictions,labels=list(range(10))).tolist()}


def save_plots(directory, metrics, history=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig,ax = plt.subplots(figsize=(7,7))
    ConfusionMatrixDisplay(np.array(metrics['confusion_matrix']),display_labels=range(10)).plot(ax=ax,colorbar=False)
    fig.tight_layout()
    fig.savefig(Path(directory)/'confusion_matrix.png',dpi=140)
    plt.close(fig)
    if history:
        fig,axes = plt.subplots(1,2,figsize=(10,4))
        for ax,key in zip(axes,['loss','accuracy']):
            for label in [key,'val_'+key]:
                ax.plot(range(1,len(history[label])+1),history[label],label=label)
            ax.set(xlabel='Epoch',ylabel=key)
            ax.legend()
        fig.tight_layout()
        fig.savefig(Path(directory)/'learning_curves.png',dpi=140)
        plt.close(fig)


def run_experiment(name, data, output, epochs=15, batch_size=64, verbose=1, **model_params):
    if epochs < 1 or batch_size < 1:
        raise ValueError('epochs and batch_size must be positive.')
    directory = Path(output)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f'{directory} already contains files. Choose a new output directory.')
    model = build_model(name,seed=data.seed,**model_params)
    directory.mkdir(parents=True,exist_ok=True)
    config = {'model':name,'seed':data.seed,'validation_size':data.validation_size,
              'epochs':epochs,'batch_size':batch_size,'model_params':model_params,
              'data_source':data.source,'n_train':len(data.y_train),'n_val':len(data.y_val),'n_test':len(data.y_test),
              'python':platform.python_version(),
              'versions':{p:importlib.metadata.version(p) for p in ['numpy','scikit-learn','tensorflow']}}
    write_json(directory/'config.json',config)
    np.savez_compressed(directory/'split_indices.npz',train=data.train_indices,val=data.val_indices,test=data.test_indices)
    start = perf_counter()
    history = {}
    if name in DEEP_MODELS:
        import tensorflow as tf
        history = model.fit(batches(model_input(data.x_train,name),data.y_train,batch_size,True,data.seed),
                            validation_data=batches(model_input(data.x_val,name),data.y_val,batch_size),
                            epochs=epochs,verbose=verbose,
                            callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.2,patience=2,min_lr=1e-6)]).history
        history = {k:[float(v) for v in values] for k,values in history.items()}
    else:
        model.fit(model_input(data.x_train,name),data.y_train)
    train_seconds = perf_counter()-start
    start = perf_counter()
    pred = raw_predictions(model,name,data.x_test,include_scores=False)
    inference_seconds = perf_counter()-start
    metrics = metrics_for(data.y_test,pred['predictions'])
    metrics.update(model=name,train_seconds=train_seconds,inference_seconds=inference_seconds,
                   ms_per_image=inference_seconds/len(data.y_test)*1000)
    save_model(model,name,directory)
    write_json(directory/'history.json',history)
    write_json(directory/'metrics.json',metrics)
    np.save(directory/'predictions.npy',pred['predictions'])
    save_plots(directory,metrics,history)
    return metrics


def predict_images(run, images):
    config = json.loads((Path(run)/'config.json').read_text(encoding='utf-8'))
    model = load_model(config['model'],run)
    return {k:np.asarray(v).tolist() for k,v in raw_predictions(model,config['model'],images).items()}


def evaluate_saved(run, images, labels):
    pred = predict_images(run,images)
    return metrics_for(labels,pred['predictions'])
