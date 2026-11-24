"""Command line interface; use python -m mnist_project --help."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .models import MODEL_NAMES, DEEP_MODELS
from .data import load_data, normalize_images
from .experiments import run_experiment, predict_images, evaluate_saved, write_json


def positive(value):
    number=int(value)
    if number<1: raise argparse.ArgumentTypeError('Must be a positive integer.')
    return number


def _data_args(parser):
    parser.add_argument('--data',type=Path,help='Local MNIST NPZ; omitted = download official MNIST.')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--validation-size',type=float,default=.1)
    parser.add_argument('--train-limit',type=positive)
    parser.add_argument('--test-limit',type=positive)


def _load(args):
    return load_data(args.data,seed=args.seed,validation_size=args.validation_size,
                     train_limit=args.train_limit,test_limit=args.test_limit)


def read_image(path, invert=False):
    from PIL import Image
    with Image.open(path) as image:
        # Preserve aspect ratio; MNIST expects centered strokes on a black canvas.
        image=image.convert('L')
        image.thumbnail((28,28),Image.Resampling.LANCZOS)
        canvas=Image.new('L',(28,28),color=255 if invert else 0)
        canvas.paste(image,((28-image.width)//2,(28-image.height)//2))
        pixels=np.asarray(canvas)
    if invert: pixels=255-pixels
    return normalize_images(pixels)


def main(argv=None):
    parser=argparse.ArgumentParser(description='MNIST: seven model families and Explainable AI')
    commands=parser.add_subparsers(dest='command',required=True)
    train=commands.add_parser('train',help='Train, save and evaluate one or all models.')
    _data_args(train)
    train.add_argument('--model',choices=(*MODEL_NAMES,'all'),default='cnn')
    train.add_argument('--config',type=Path,help='JSON training defaults.')
    train.add_argument('--params',type=Path,help='best_params.json from tune, or a model parameter JSON.')
    train.add_argument('--epochs',type=positive)
    train.add_argument('--batch-size',type=positive)
    train.add_argument('--output',type=Path,help='New run directory; all models get separate subdirectories.')
    tune=commands.add_parser('tune',help='Optuna on training/validation only.')
    _data_args(tune)
    tune.add_argument('--model',choices=DEEP_MODELS,required=True)
    tune.add_argument('--trials',type=positive,default=5)
    tune.add_argument('--epochs',type=positive,default=10)
    tune.add_argument('--output',type=Path,required=True)
    eda=commands.add_parser('eda',help='Export distribution, samples, mean images and sparsity plots.')
    _data_args(eda); eda.add_argument('--output',type=Path,required=True)
    evaluate=commands.add_parser('evaluate',help='Evaluate saved model on the recorded official test subset.')
    evaluate.add_argument('--run',type=Path,required=True)
    evaluate.add_argument('--data',type=Path,help='Override original NPZ location.')
    evaluate.add_argument('--output',type=Path,help='New JSON report path.')
    for command in ['predict','explain']:
        sub=commands.add_parser(command)
        sub.add_argument('--run',type=Path,required=True)
        source=sub.add_mutually_exclusive_group(required=True)
        source.add_argument('--image',type=Path,help='28x28 centered light digit on dark background.')
        source.add_argument('--images-npy',type=Path,help='uint8 or normalized NumPy image batch; explain uses first image.')
        sub.add_argument('--invert',action='store_true',help='Invert an image file with a white background.')
        sub.add_argument('--output',type=Path,required=command=='explain')
        if command=='explain':
            sub.add_argument('--method',choices=['auto','neighbors','importance','occlusion','saliency','ig','gradcam','attention'],default='auto')
            sub.add_argument('--target',type=int,choices=range(10))
            sub.add_argument('--steps',type=positive,default=32)
    args=parser.parse_args(argv)
    try:
        if args.command=='train':
            defaults=json.loads(args.config.read_text(encoding='utf-8')) if args.config else {}
            params=json.loads(args.params.read_text(encoding='utf-8')) if args.params else {}
            if 'best_params' in params:
                if params['model']!=args.model: raise ValueError('Tuning parameters belong to a different model.')
                params=params['best_params']
            tuned_batch=params.pop('batch_size',None)
            batch=args.batch_size or tuned_batch or defaults.get('batch_size',64)
            epochs=args.epochs or defaults.get('epochs',15)
            allowed={'learning_rate','num_filters','dropout_rate'}
            if set(params)-allowed: raise ValueError(f'Unknown model parameters: {set(params)-allowed}')
            if args.model=='all' and params: raise ValueError('Apply tuning parameters to one model at a time.')
            output=args.output or Path('artifacts')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            if output.exists() and any(output.iterdir()): raise FileExistsError(f'{output} already contains files.')
            data=_load(args)
            results=[]
            for name in MODEL_NAMES if args.model=='all' else [args.model]:
                print(f'Training {name} ...',flush=True)
                results.append(run_experiment(name,data,output/name if args.model=='all' else output,
                                               epochs=epochs,batch_size=batch,**params))
            fields=['model','accuracy','f1_macro','f1_weighted','train_seconds','inference_seconds','ms_per_image','n_test']
            with (output/'comparison.csv').open('w',newline='',encoding='utf-8') as file:
                writer=csv.DictWriter(file,fieldnames=fields,extrasaction='ignore')
                writer.writeheader(); writer.writerows(results)
            print(f'Artifacts: {output.resolve()}')
        elif args.command=='tune':
            from .tuning import tune
            print(json.dumps(tune(args.model,_load(args),args.output,args.trials,args.epochs),indent=2))
        elif args.command=='eda':
            from .eda import create_eda
            create_eda(_load(args),args.output)
            print(f'EDA: {args.output.resolve()}')
        elif args.command=='evaluate':
            config=json.loads((args.run/'config.json').read_text(encoding='utf-8'))
            source=args.data or Path(config['data_source'])
            with np.load(source,allow_pickle=False) as raw:
                indices=np.load(args.run/'split_indices.npz',allow_pickle=False)['test']
                result=evaluate_saved(args.run,raw['x_test'][indices],raw['y_test'][indices])
            if args.output:
                if args.output.exists(): raise FileExistsError(f'{args.output} exists.')
                write_json(args.output,result)
            print(json.dumps(result,indent=2))
        else:
            images=read_image(args.image,args.invert) if args.image else normalize_images(np.load(args.images_npy,allow_pickle=False))
            if args.command=='predict':
                result=predict_images(args.run,images)
                if args.output:
                    if args.output.exists(): raise FileExistsError(f'{args.output} exists.')
                    write_json(args.output,result)
                print(json.dumps(result,indent=2))
            else:
                from .explain import explain_run
                explain_run(args.run,images[:1],args.output,args.method,args.target,args.steps)
                print(f'Explanation: {args.output.resolve()}')
    except (ValueError,FileNotFoundError,FileExistsError,KeyError) as exc:
        parser.exit(2,f'Error: {exc}\n')


if __name__=='__main__': main()
