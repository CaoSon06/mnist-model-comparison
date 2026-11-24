"""Factories derived from AI-Final.ipynb; no training at import time."""
from pathlib import Path
import joblib
import numpy as np

ML_MODELS = ('knn','svm','random_forest')
DEEP_MODELS = ('mlp','cnn','resnet9','tinyvit')
MODEL_NAMES = ML_MODELS + DEEP_MODELS


def build_model(name, seed=42, learning_rate=None, num_filters=32, dropout_rate=None):
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}.")
    if name == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    if name == 'svm':
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        return make_pipeline(StandardScaler(), SVC(C=.5, kernel='rbf', random_state=seed))
    if name == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5,
                                      random_state=seed, n_jobs=-1)
    import tensorflow as tf
    from tensorflow.keras import layers as L
    tf.keras.utils.set_random_seed(seed)
    rate = learning_rate if learning_rate is not None else {'resnet9': .003117802499283948, 'tinyvit': .0004511621607310748}.get(name, .001)
    drop = dropout_rate if dropout_rate is not None else {'resnet9': .22150869266027387, 'tinyvit': .25444330590087644}.get(name, .3)

    def dense_head(x):
        x = L.Flatten()(x)
        x = L.Dense(128)(x)
        x = L.BatchNormalization()(x)
        x = L.Activation('relu')(x)
        return L.Dropout(drop)(x)

    def conv(x, filters, prefix):
        x = L.Conv2D(filters, 3, padding='same', name=prefix+'_conv')(x)
        x = L.BatchNormalization(name=prefix+'_bn')(x)
        return L.Activation('relu', name=prefix+'_activation')(x)

    inputs = L.Input((784,) if name=='mlp' else (28,28,1), name='images')
    if name=='mlp':
        x = L.Dense(num_filters)(inputs)
        x = L.BatchNormalization()(x)
        x = L.Activation('relu')(x)
        x = dense_head(x)
    elif name=='cnn':
        x = L.MaxPooling2D()(conv(inputs, num_filters, 'block1'))
        x = L.MaxPooling2D()(conv(x, 64, 'target'))
        x = dense_head(x)
    elif name=='resnet9':
        x = L.MaxPooling2D()(conv(inputs, num_filters, 'init'))
        for i, filters in enumerate([num_filters,64],1):
            shortcut = x
            y = conv(x, filters, f'res{i}')
            y = L.Conv2D(filters,3,padding='same')(y)
            y = L.BatchNormalization()(y)
            if shortcut.shape[-1] != filters:
                shortcut = L.Conv2D(filters,1,padding='same')(shortcut)
                shortcut = L.BatchNormalization()(shortcut)
            x = L.Add()([y,shortcut])
            x = L.Activation('relu',name=f'res{i}_out')(x)
        x = L.MaxPooling2D()(x)
        x = dense_head(x)
    else:
        from .vit import PositionEmbedding
        x = L.Conv2D(64,4,strides=4,name='patch_conv')(inputs)
        x = L.Reshape((49,64))(x)
        x = PositionEmbedding(name='position_embedding')(x)
        for i in range(2):
            norm = L.LayerNormalization(epsilon=1e-6, name=f'attention_input_{i}')(x)
            attention = L.MultiHeadAttention(num_heads=4,key_dim=64,name=f'attention_{i}')(norm,norm)
            x = L.Add()([x,L.Dropout(drop)(attention)])
            norm = L.LayerNormalization(epsilon=1e-6)(x)
            mlp = L.Dense(128,activation='gelu')(norm)
            mlp = L.Dropout(drop)(mlp)
            mlp = L.Dense(64,activation='gelu')(mlp)
            x = L.Add()([x,mlp])
        x = L.LayerNormalization(epsilon=1e-6)(x)
        x = L.GlobalAveragePooling1D()(x)
        x = L.Dropout(drop)(x)
    outputs = L.Dense(10,activation='softmax',name='output_layer')(x)
    model = tf.keras.Model(inputs,outputs,name=name)
    model.compile(optimizer=tf.keras.optimizers.Adam(rate),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model


def save_model(model, name, directory):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    if name in DEEP_MODELS:
        model.save(directory/'model.keras')
    else:
        joblib.dump(model,directory/'model.joblib')


def load_model(name, directory):
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}")
    directory = Path(directory)
    if name in DEEP_MODELS:
        import tensorflow as tf
        from . import vit  # Registers the serialized position embedding.
        return tf.keras.models.load_model(directory/'model.keras')
    return joblib.load(directory/'model.joblib')
