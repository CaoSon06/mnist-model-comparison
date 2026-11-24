"""Shared preprocessing and strictly disjoint train/validation/test splits."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split


@dataclass
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    seed: int
    validation_size: float
    source: str = "provided arrays"


def normalize_images(images):
    """Accept uint8 [0,255] or floating [0,1]; return (N,28,28,1)."""
    arr = np.asarray(images)
    if arr.ndim == 2 and arr.shape == (28, 28):
        arr = arr[None, ..., None]
    elif arr.ndim == 3 and arr.shape[1:] == (28,28):
        arr = arr[..., None]
    elif arr.ndim == 3 and arr.shape == (28,28,1):
        arr = arr[None]
    if arr.ndim != 4 or arr.shape[1:] != (28,28,1) or len(arr) == 0:
        raise ValueError("Images must have shape (N,28,28), (N,28,28,1), or one 28x28 image.")
    if arr.dtype.kind not in 'uif' or not np.isfinite(arr).all():
        raise ValueError("Images must contain finite numeric pixels.")
    high = 255 if arr.dtype.kind in 'ui' else 1
    if arr.min() < 0 or arr.max() > high:
        raise ValueError("Integer pixels must be [0,255]; floating pixels must be [0,1].")
    return arr.astype(np.float32) / (255. if arr.dtype.kind in 'ui' else 1.)


def _labels(values, count):
    labels = np.asarray(values)
    if labels.shape != (count,) or labels.dtype.kind not in 'ui' or np.any((labels < 0) | (labels > 9)):
        raise ValueError("Labels must be a one-dimensional integer array of digits 0–9 matching images.")
    return labels.astype(np.int64)


def _limit(indices, labels, limit, seed):
    if limit is None:
        return indices
    if limit < len(np.unique(labels[indices])):
        raise ValueError("Subset limit must allow at least one image per available class.")
    if limit >= len(indices):
        return indices
    classes, counts = np.unique(labels[indices],return_counts=True)
    # Reserve one example per class, then allocate the remaining budget
    # proportionally with largest remainders. The discarded side need not
    # contain every class (unlike train_test_split's stratification).
    remaining=limit-len(classes)
    capacity=counts-1
    quotas=remaining*capacity/capacity.sum()
    extra=np.floor(quotas).astype(int)
    remainder=remaining-int(extra.sum())
    extra[np.argsort(-(quotas-extra),kind='stable')[:remainder]]+=1
    rng=np.random.default_rng(seed)
    selected=np.concatenate([rng.choice(indices[labels[indices]==label],int(count),replace=False)
                             for label,count in zip(classes,extra+1)])
    rng.shuffle(selected)
    return selected


def split_data(x_train, y_train, x_test, y_test, validation_size=.1, seed=42,
               train_limit=None, test_limit=None, source="provided arrays"):
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")
    # Split/sample uint8 arrays before conversion to keep peak memory modest.
    x_train, x_test = np.asarray(x_train), np.asarray(x_test)
    y_train, y_test = _labels(y_train, len(x_train)), _labels(y_test, len(x_test))
    train_idx, val_idx = train_test_split(np.arange(len(y_train)), test_size=validation_size,
                                         stratify=y_train, random_state=seed)
    train_idx = _limit(train_idx, y_train, train_limit, seed)
    test_idx = _limit(np.arange(len(y_test)), y_test, test_limit, seed)
    return Dataset(normalize_images(x_train[train_idx]), y_train[train_idx],
                   normalize_images(x_train[val_idx]), y_train[val_idx],
                   normalize_images(x_test[test_idx]), y_test[test_idx],
                   train_idx, val_idx, test_idx, seed, validation_size, source)


def load_data(path=None, **kwargs):
    """Load a local Keras-style NPZ, or download MNIST into ignored data/."""
    if path is None:
        from tensorflow.keras.utils import get_file
        path = get_file('mnist.npz', origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz',
                        file_hash='731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1',
                        cache_dir=str(Path('data').resolve()), cache_subdir='raw')
    with np.load(Path(path), allow_pickle=False) as archive:
        arrays = [archive[key] for key in ['x_train','y_train','x_test','y_test']]
    return split_data(*arrays, source=str(Path(path).resolve()), **kwargs)


def model_input(images, name):
    images = normalize_images(images)
    return images.reshape(len(images), -1) if name in {'knn','svm','random_forest','mlp'} else images
