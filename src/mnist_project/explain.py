"""Model-specific explanations. Gradient attribution uses pre-softmax logits."""
import numpy as np
from .data import normalize_images, model_input


def _image(image):
    images = normalize_images(image)
    if len(images)!=1:
        raise ValueError('Explain exactly one image at a time.')
    return images


def _target(model, image, target):
    values = model(image,training=False).numpy()[0]
    target = int(values.argmax()) if target is None else int(target)
    if not 0<=target<10:
        raise ValueError('Target digit must be 0–9.')
    return target


def _logits(model):
    import tensorflow as tf
    final = model.layers[-1]
    linear = tf.keras.layers.Dense(final.units,use_bias=final.use_bias)
    output = linear(final.input)
    result = tf.keras.Model(model.input,output)
    linear.set_weights(final.get_weights())
    return result


def _input(model, images):
    return images.reshape(len(images),784) if len(model.input_shape)==2 else images


def saliency(model, image, target=None):
    import tensorflow as tf
    image = _input(model,_image(image))
    target = _target(model,image,target)
    inputs = tf.convert_to_tensor(image)
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        scores = _logits(model)(inputs,training=False)[:,target]
    return np.abs(tape.gradient(scores,inputs).numpy()).reshape(28,28)


def integrated_gradients(model, image, target=None, steps=32):
    """Signed attribution with black baseline and trapezoidal integration."""
    import tensorflow as tf
    if steps<1:
        raise ValueError('steps must be positive.')
    image = _input(model,_image(image))
    target = _target(model,image,target)
    alphas = np.linspace(0,1,steps+1,dtype=np.float32).reshape((-1,)+(1,)*(image.ndim-1))
    interpolated = tf.convert_to_tensor(alphas*image)
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        score = _logits(model)(interpolated,training=False)[:,target]
    gradients = tape.gradient(score,interpolated).numpy()
    mean_grad = ((gradients[:-1]+gradients[1:])/2).mean(axis=0)
    return (image[0]*mean_grad).reshape(28,28)


def _unit(values):
    values = np.asarray(values,dtype=np.float32)
    maximum = np.max(np.abs(values))
    return values/maximum if maximum>0 else np.zeros_like(values)


def gradcam(model, image, target=None):
    import tensorflow as tf
    image = _image(image)
    target = _target(model,image,target)
    layer = 'target_activation' if model.name=='cnn' else 'res2_out'
    if model.name not in {'cnn','resnet9'}:
        raise ValueError('Grad-CAM supports cnn and resnet9.')
    logits = _logits(model)
    extractor = tf.keras.Model(model.input,[model.get_layer(layer).output,logits.output])
    with tf.GradientTape() as tape:
        features,scores = extractor(image,training=False)
        score = scores[:,target]
    gradients = tape.gradient(score,features)
    weights = tf.reduce_mean(gradients,axis=(1,2),keepdims=True)
    heatmap = tf.nn.relu(tf.reduce_sum(weights*features,axis=-1))[0]
    return _unit(tf.image.resize(heatmap[...,None],(28,28)).numpy().squeeze())


def attention_rollout(model, image):
    import tensorflow as tf
    if model.name!='tinyvit':
        raise ValueError('Attention rollout requires tinyvit.')
    image = _image(image)
    rollout = np.eye(49,dtype=np.float32)
    for i in range(2):
        extractor = tf.keras.Model(model.input,model.get_layer(f'attention_input_{i}').output)
        tokens = extractor(image,training=False)
        _,scores = model.get_layer(f'attention_{i}')(tokens,tokens,return_attention_scores=True,training=False)
        attention = scores.numpy()[0].mean(axis=0)+np.eye(49)
        attention /= attention.sum(axis=-1,keepdims=True)
        rollout = attention@rollout
    heatmap = rollout.mean(axis=0).reshape(7,7)
    heatmap -= heatmap.min()
    return _unit(tf.image.resize(heatmap[...,None],(28,28)).numpy().squeeze())


def svm_occlusion(model, image, patch_size=4, stride=2, target=None):
    """Change in decision score, not a calibrated probability."""
    image = _image(image)
    if not 1<=patch_size<=28 or stride<1:
        raise ValueError('patch_size must be 1–28 and stride must be positive.')
    scores = model.decision_function(model_input(image,'svm'))[0]
    target = int(np.argmax(scores)) if target is None else int(np.where(model.classes_==target)[0][0])
    positions = sorted(set(range(0,29-patch_size,stride))|{28-patch_size})
    locations = [(r,c) for r in positions for c in positions]
    masked = np.repeat(image,len(locations),axis=0)
    for sample,(row,col) in zip(masked,locations):
        sample[row:row+patch_size,col:col+patch_size]=0
    drops = scores[target]-model.decision_function(model_input(masked,'svm'))[:,target]
    heat,counts = np.zeros((28,28)),np.zeros((28,28))
    for (row,col),drop in zip(locations,drops):
        heat[row:row+patch_size,col:col+patch_size]+=drop
        counts[row:row+patch_size,col:col+patch_size]+=1
    return heat/np.maximum(counts,1)


def deletion_test(model, image, attribution, fractions=(0,.05,.1,.2,.3), random_runs=5, seed=42, target=None):
    """Single-image illustration, not dataset-wide evidence of faithfulness."""
    image = _image(image)
    if random_runs<1 or any(not 0<=f<=1 for f in fractions):
        raise ValueError('Use positive random_runs and fractions in [0,1].')
    target = _target(model,_input(model,image),target)
    def probability(x): return float(model(_input(model,x),training=False).numpy()[0,target])
    baseline = probability(image)
    order = np.argsort(np.abs(attribution).ravel())[::-1]
    rng = np.random.default_rng(seed)
    rows = []
    for fraction in fractions:
        k = int(fraction*784)
        guided = image.copy().reshape(-1)
        guided[order[:k]]=0
        random_scores=[]
        for _ in range(random_runs):
            random_image=image.copy().reshape(-1)
            random_image[rng.choice(784,k,replace=False)]=0
            random_scores.append(probability(random_image.reshape(image.shape)))
        rows.append({'target':target,'fraction':fraction,'guided_drop':baseline-probability(guided.reshape(image.shape)),
                     'random_drop':baseline-float(np.mean(random_scores))})
    return rows


def explain_run(run, image, output, method='auto', target=None, steps=32):
    import json
    from pathlib import Path
    from .models import load_model, DEEP_MODELS
    from .experiments import write_json
    config = json.loads((Path(run)/'config.json').read_text(encoding='utf-8'))
    name = config['model']
    model = load_model(name,run)
    image = _image(image)
    if method=='auto':
        method = {'knn':'neighbors','svm':'occlusion','random_forest':'importance','mlp':'ig',
                  'cnn':'gradcam','resnet9':'gradcam','tinyvit':'attention'}[name]
    directory=Path(output)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f'{directory} already contains files.')
    directory.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metadata={'model':name,'method':method,'requested_target':target}
    if method=='neighbors' and name=='knn':
        distances,indices = model.kneighbors(model_input(image,name))
        maps = model._fit_X[indices[0]].reshape(-1,28,28)
        # _y is encoded in KNeighborsClassifier; map it back to original digits.
        labels = model.classes_[model._y[indices[0]]]
        metadata.update(distances=distances[0].tolist(),labels=labels.tolist())
        fig,axes=plt.subplots(1,len(maps)+1,figsize=(3*(len(maps)+1),3))
        axes[0].imshow(image.squeeze(),cmap='gray'); axes[0].set_title('Input')
        for ax,values,label in zip(axes[1:],maps,labels):
            ax.imshow(values,cmap='gray'); ax.set_title(f'Neighbor: {label}')
    else:
        if name in DEEP_MODELS and method in {'ig','saliency','gradcam','attention'}:
            call={'ig':integrated_gradients,'saliency':saliency,'gradcam':gradcam,'attention':attention_rollout}[method]
            kwargs={} if method=='attention' else {'target':target}
            if method=='ig': kwargs['steps']=steps
            importance=call(model,image,**kwargs)
            metadata['class_specific']=method!='attention'
            metadata['target']=None if method=='attention' else _target(model,_input(model,image),target)
            if method=='ig': metadata['deletion_test']=deletion_test(model,image,importance,target=metadata['target'])
        elif method=='importance' and name=='random_forest':
            importance=model.feature_importances_.reshape(28,28)
        elif method=='occlusion' and name=='svm':
            importance=svm_occlusion(model,image,target=target)
        else:
            raise ValueError(f'Method {method} is not supported for {name}.')
        np.save(directory/'attribution.npy',importance)
        fig,axes=plt.subplots(1,3,figsize=(10,3))
        axes[0].imshow(image.squeeze(),cmap='gray'); axes[0].set_title('Input')
        axes[1].imshow(_unit(importance),cmap='coolwarm',vmin=-1,vmax=1); axes[1].set_title(method)
        axes[2].imshow(image.squeeze(),cmap='gray')
        axes[2].imshow(np.abs(_unit(importance)),cmap='magma',alpha=.6,vmin=0,vmax=1)
        axes[2].set_title('Overlay')
    for ax in axes: ax.axis('off')
    fig.tight_layout(); fig.savefig(directory/'explanation.png',dpi=150); plt.close(fig)
    write_json(directory/'explanation.json',metadata)
    return metadata
