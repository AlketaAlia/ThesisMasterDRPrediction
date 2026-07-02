"""Shared utilities for DR prediction training and evaluation.

The helpers here are designed so that:
- Train and test data generators are *separate* — augmentation is applied to
  training data only, never to the held-out test set.
- The train/test split is stratified by label, so class proportions are
  preserved across splits.
- Per-architecture preprocessing functions (ImageNet `preprocess_input`) can be
  passed in, keeping training and inference consistent.
"""
import logging
import os
import pickle

import cv2
import numpy as np
import pandas as pd
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator


logger = logging.getLogger(__name__)


def get_labels(labels_path):
    labels = pd.read_csv(labels_path)
    labels.rename(columns={'id_code': 'filename', 'diagnosis': 'label'}, inplace=True)
    return labels


def num_to_diagnosis(val):
    array = ['no_dr', 'mild', 'moderate', 'severe', 'proliferate']
    if 0 <= val < 5:
        return array[int(val)]
    raise ValueError(f'val must be in [0, 5), got {val}')


def read_all_images(image_dir_path, use_pickle=True):
    """Load all PNG images from a directory into a DataFrame.

    Caches the result to a pickle file. If the cached filepaths are stale
    (e.g., the project was moved to a new machine), the cache is rebuilt.
    """
    pickle_file = os.path.join(image_dir_path, 'all_images.pcl')
    if use_pickle and os.path.isfile(pickle_file):
        with open(pickle_file, 'rb') as file:
            all_images = pickle.load(file)
        if (
            isinstance(all_images, pd.DataFrame)
            and 'filepath' in all_images.columns
            and all_images['filepath'].head(10).apply(os.path.isfile).all()
        ):
            return all_images
        logger.info('Cached image paths are stale, rebuilding.')

    all_images = []
    for filename in sorted(os.listdir(image_dir_path)):
        if filename.lower().endswith('.png'):
            image = io.imread(os.path.join(image_dir_path, filename))
            all_images.append({
                'filepath': os.path.join(image_dir_path, filename),
                'filename': os.path.splitext(filename)[0],
                'image': image,
            })
    all_images = pd.DataFrame.from_dict(all_images)
    with open(pickle_file, 'wb') as file:
        pickle.dump(all_images, file)
    return all_images


def check_image_sizes(images):
    sizes = {tuple(row['image'].shape) for _, row in images.iterrows()}
    if len(sizes) == 1:
        shape = next(iter(sizes))
        logger.info('All images share the same size: %s', shape)
        return {'width': shape[0], 'height': shape[1], 'depth': shape[2]}
    logger.warning('Images have different sizes: %s', sizes)
    return sizes


def build_input_df(images, labels, multiclass=False):
    """Merge images with labels and prepare the target column.

    Args:
        multiclass: if False (default, thesis-bachelor formulation), binarize
            DR severity 0 → No DR (0), severities 1-4 → DR (1). If True, keep
            the 5-class severity scale (0=No DR, 1=Mild, 2=Moderate, 3=Severe,
            4=PDR) — phase-3 master extension.
    """
    data = pd.merge(images, labels, on='filename', how='inner')
    if not multiclass:
        data.loc[data['label'] >= 1, 'label'] = 1
    data['label'] = data['label'].astype(str)
    return data


# Human-readable names for the 5-class severity scale, matching the thesis
# Section 4.1 / APTOS 2019 grading.
DR_CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "PDR"]


def grayscale_conversion(img):
    """Convert RGB image to grayscale and replicate to 3 channels.

    skimage.io.imread loads images as RGB, so we use COLOR_RGB2GRAY (the
    original code used COLOR_BGR2GRAY which gives slightly different weights).

    Returns float32 — Keras's ImageDataGenerator applies rescale=1/255 in
    place after this function runs, and `uint8 *= 1/255` raises a casting
    error in newer numpy.
    """
    grayscale_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return np.stack((grayscale_img,) * 3, axis=-1).astype(np.float32)


def stratified_split(data, test_size=0.2, val_size=0.1, random_state=123):
    """Stratified train/val/test split.

    Returns (train, val, test) DataFrames. `val_size` is a fraction of the
    *original* dataset, taken from the training portion.
    """
    train_val, test = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data['label'],
    )
    if val_size <= 0:
        return train_val, None, test
    relative_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        random_state=random_state,
        stratify=train_val['label'],
    )
    return train, val, test


def compute_balanced_class_weights(labels):
    """Compute class weights inversely proportional to class frequency.

    Works for both binary and multi-class labels (any integer K classes).
    Pass the result to `model.fit(class_weight=...)` to handle imbalance
    without resampling.
    """
    y = np.asarray(labels).astype(int)
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    return dict(zip(classes.tolist(), weights.tolist()))


def build_train_datagen(preprocessing_function=None, use_grayscale=False):
    """Augmenting data generator for the *training* set only.

    Args:
        preprocessing_function: Optional architecture-specific preprocessor
            (e.g., `tf.keras.applications.resnet50.preprocess_input`). If
            provided, it replaces simple `rescale=1/255` — they should not be
            combined.
        use_grayscale: Whether to convert images to grayscale before further
            preprocessing. Set to True only for the from-scratch CNN that was
            trained on grayscale-replicated inputs.
    """
    if preprocessing_function is not None and use_grayscale:
        def combined(img):
            return preprocessing_function(grayscale_conversion(img))
        preproc = combined
        rescale = None
    elif use_grayscale:
        preproc = grayscale_conversion
        rescale = 1.0 / 255
    elif preprocessing_function is not None:
        preproc = preprocessing_function
        rescale = None
    else:
        preproc = None
        rescale = 1.0 / 255

    return ImageDataGenerator(
        rescale=rescale,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        preprocessing_function=preproc,
    )


def build_eval_datagen(preprocessing_function=None, use_grayscale=False):
    """Non-augmenting data generator for validation and test sets.

    Mirrors the preprocessing of `build_train_datagen` but without rotation,
    flips, shifts, etc. — held-out evaluation must run on unmodified images.
    """
    if preprocessing_function is not None and use_grayscale:
        def combined(img):
            return preprocessing_function(grayscale_conversion(img))
        preproc = combined
        rescale = None
    elif use_grayscale:
        preproc = grayscale_conversion
        rescale = 1.0 / 255
    elif preprocessing_function is not None:
        preproc = preprocessing_function
        rescale = None
    else:
        preproc = None
        rescale = 1.0 / 255

    return ImageDataGenerator(rescale=rescale, preprocessing_function=preproc)


# Backwards compatibility for any code still importing the old generator
def build_data_generator():
    """Deprecated: use build_train_datagen / build_eval_datagen instead."""
    logger.warning(
        'build_data_generator() is deprecated; use build_train_datagen() and '
        'build_eval_datagen() to keep test data un-augmented.'
    )
    return build_train_datagen(use_grayscale=True)
