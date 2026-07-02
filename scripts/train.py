"""Unified training script for all DR prediction models.

Replaces the five near-identical scripts (cnn.py, ResNet50.py, DenseNet121.py,
VGG16.py, Xception.py) with a single CLI-driven entry point.

Usage:
    python train.py --arch resnet50
    python train.py --arch xception --epochs 50 --finetune
    python train.py --arch cnn --epochs 100
    python train.py --arch vgg_scratch
"""
import argparse
import logging
import os
import pickle
from dataclasses import dataclass
from typing import Callable, Optional

from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    DenseNet121, ResNet50, VGG16, Xception,
)
from tensorflow.keras.applications import densenet, resnet50, vgg16, xception
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam

from helpers import (
    build_eval_datagen,
    build_input_df,
    build_train_datagen,
    check_image_sizes,
    compute_balanced_class_weights,
    get_labels,
    read_all_images,
    stratified_split,
)
from visualization import plot_nn_accuracy, plot_nn_loss


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('train')


@dataclass
class ArchSpec:
    name: str
    model_filename: str
    history_filename: str
    weights_filename: Optional[str]
    builder: Callable
    preprocess: Optional[Callable]
    use_grayscale: bool


def _build_transfer(base_class, weights_path, image_size, num_classes=1):
    """Build a transfer-learning model with a frozen base + small head.

    Args:
        num_classes: 1 → binary head (sigmoid). >1 → multi-class softmax head.
    """
    weights = weights_path if weights_path and os.path.isfile(weights_path) else 'imagenet'
    base = base_class(weights=weights, include_top=False, input_shape=(*image_size, 3))
    for layer in base.layers:
        layer.trainable = False
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    if num_classes == 1:
        out = layers.Dense(1, activation='sigmoid')(x)
    else:
        out = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=base.input, outputs=out), base


def _build_cnn_from_scratch(image_size, num_classes=1):
    """Original 3-block CNN, kept for reproducibility of thesis results.

    `num_classes=1` gives a binary sigmoid head; `>1` gives a multi-class
    softmax head over `num_classes` outputs.
    """
    final_units = num_classes if num_classes > 1 else 1
    final_act = "softmax" if num_classes > 1 else "sigmoid"
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*image_size, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(final_units, activation=final_act),
    ])
    return model, None


def _build_cnn_mc_dropout(image_size, dropout_rate=0.3):
    """3-block CNN with dropout layers always-active for MC sampling.

    Standard dropout is disabled at inference. For Monte Carlo Dropout, we
    leave it on so multiple forward passes give different predictions —
    the variance across passes approximates Bayesian model uncertainty
    (Gal & Ghahramani 2016).

    Architecture mirrors `_build_cnn_from_scratch` but inserts SpatialDropout2D
    after each conv block and Dropout before the final classifier.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*image_size, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.SpatialDropout2D(dropout_rate),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.SpatialDropout2D(dropout_rate),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.SpatialDropout2D(dropout_rate),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid'),
    ])
    return model, None


def _build_resnet_mc_dropout(image_size, dropout_rate=0.3, weights_path=None):
    """ResNet50 transfer + MC dropout in the head.

    Base is frozen ImageNet weights; we add Dropout before the classifier
    so MC sampling produces variance from the trainable head only. This is
    a "Bayesian last layer" approximation but cheap and effective.
    """
    weights = weights_path if weights_path and os.path.isfile(weights_path) else 'imagenet'
    base = ResNet50(weights=weights, include_top=False, input_shape=(*image_size, 3))
    for layer in base.layers:
        layer.trainable = False
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inputs=base.input, outputs=out), base


def _build_cnn_tanh_relu(image_size):
    """Same shape as `_build_cnn_from_scratch` but with the second conv block
    and the dense head using tanh instead of ReLU. Matches the thesis's
    "CNN(Tanh+ReLU)" variant in Section 4.4.1 / Table 9.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*image_size, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='tanh'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='tanh'),
        layers.Dense(1, activation='sigmoid'),
    ])
    return model, None


def _build_vgg_scratch(image_size):
    """Full VGG16 architecture trained from scratch (no transfer)."""
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(*image_size, 3)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dense(4096, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])
    return model, None


def get_arch_spec(name, inputs_dir, image_size):
    """Return the build function and preprocessing for the requested arch."""
    name = name.lower()
    if name == 'resnet50':
        return ArchSpec(
            name='resnet50',
            model_filename='resnet_model.keras',
            history_filename='resnet_model_history.pcl',
            weights_filename='resnet50_weights.h5',
            builder=lambda: _build_transfer(
                ResNet50, os.path.join(inputs_dir, 'resnet50_weights.h5'), image_size),
            preprocess=resnet50.preprocess_input,
            use_grayscale=False,
        )
    if name == 'densenet121':
        return ArchSpec(
            name='densenet121',
            model_filename='densenet_model.keras',
            history_filename='densenet_model_history.pcl',
            weights_filename='densenet_weights.h5',
            builder=lambda: _build_transfer(
                DenseNet121, os.path.join(inputs_dir, 'densenet_weights.h5'), image_size),
            preprocess=densenet.preprocess_input,
            use_grayscale=False,
        )
    if name == 'xception':
        return ArchSpec(
            name='xception',
            model_filename='xception_model.keras',
            history_filename='xception_model_history.pcl',
            weights_filename='xception_weights.h5',
            builder=lambda: _build_transfer(
                Xception, os.path.join(inputs_dir, 'xception_weights.h5'), image_size),
            preprocess=xception.preprocess_input,
            use_grayscale=False,
        )
    if name == 'vgg16':
        return ArchSpec(
            name='vgg16',
            model_filename='model_vgg16.keras',
            history_filename='model_history_vgg16.pcl',
            weights_filename='vgg_weights.h5',
            builder=lambda: _build_transfer(
                VGG16, os.path.join(inputs_dir, 'vgg_weights.h5'), image_size),
            preprocess=vgg16.preprocess_input,
            use_grayscale=False,
        )
    if name == 'cnn':
        return ArchSpec(
            name='cnn',
            # Use cnn_* filenames (not vgg_* like the original cnn.py) so the
            # disk filename matches what the model actually is. The original
            # naming was a vestige of when cnn.py started as a VGG transfer
            # script before being rewritten as a from-scratch CNN.
            model_filename='cnn_model.keras',
            history_filename='cnn_model_history.pcl',
            weights_filename=None,
            builder=lambda: _build_cnn_from_scratch(image_size),
            preprocess=None,
            use_grayscale=True,
        )
    if name == 'cnn_5class':
        return ArchSpec(
            name='cnn_5class',
            model_filename='cnn_5class_model.keras',
            history_filename='cnn_5class_model_history.pcl',
            weights_filename=None,
            builder=lambda: _build_cnn_from_scratch(image_size, num_classes=5),
            preprocess=None,
            use_grayscale=True,
        )
    if name == 'resnet50_5class':
        return ArchSpec(
            name='resnet50_5class',
            model_filename='resnet50_5class_model.keras',
            history_filename='resnet50_5class_model_history.pcl',
            weights_filename='resnet50_weights.h5',
            builder=lambda: _build_transfer(
                ResNet50, os.path.join(inputs_dir, 'resnet50_weights.h5'),
                image_size, num_classes=5),
            preprocess=resnet50.preprocess_input,
            use_grayscale=False,
        )
    if name == 'cnn_mcd':
        return ArchSpec(
            name='cnn_mcd',
            model_filename='cnn_mcd_model.keras',
            history_filename='cnn_mcd_model_history.pcl',
            weights_filename=None,
            builder=lambda: _build_cnn_mc_dropout(image_size),
            preprocess=None,
            use_grayscale=True,
        )
    if name == 'resnet50_mcd':
        return ArchSpec(
            name='resnet50_mcd',
            model_filename='resnet_mcd_model.keras',
            history_filename='resnet_mcd_model_history.pcl',
            weights_filename='resnet50_weights.h5',
            builder=lambda: _build_resnet_mc_dropout(
                image_size,
                weights_path=os.path.join(inputs_dir, 'resnet50_weights.h5')),
            preprocess=resnet50.preprocess_input,
            use_grayscale=False,
        )
    if name == 'cnn_tanh':
        return ArchSpec(
            name='cnn_tanh',
            model_filename='cnn_tanh_model.keras',
            history_filename='cnn_tanh_model_history.pcl',
            weights_filename=None,
            builder=lambda: _build_cnn_tanh_relu(image_size),
            preprocess=None,
            use_grayscale=True,
        )
    if name == 'vgg_scratch':
        return ArchSpec(
            name='vgg_scratch',
            model_filename='vgg_scratch_model.keras',
            history_filename='vgg_scratch_history.pcl',
            weights_filename=None,
            builder=lambda: _build_vgg_scratch(image_size),
            preprocess=None,
            use_grayscale=False,
        )
    raise ValueError(f'Unknown arch: {name!r}')


def make_callbacks(model_path, patience=10):
    """EarlyStopping + ReduceLROnPlateau + ModelCheckpoint(best)."""
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=max(3, patience // 2),
            min_lr=1e-6,
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
    ]


def fine_tune(model, base_model, generator_train, generator_val, epochs,
              callbacks, class_weight, loss_fn='binary_crossentropy',
              unfreeze_from=-30, learning_rate=1e-5):
    """Unfreeze the top layers of `base_model` and continue training.

    For transfer-learning architectures only — pass `base_model=None` to skip.
    """
    if base_model is None:
        logger.info('Architecture has no base model to fine-tune; skipping.')
        return None
    logger.info('Fine-tuning: unfreezing last %d layers', abs(unfreeze_from))
    for layer in base_model.layers[unfreeze_from:]:
        layer.trainable = True
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=['accuracy'],
    )
    return model.fit(
        generator_train,
        epochs=epochs,
        validation_data=generator_val,
        callbacks=callbacks,
        class_weight=class_weight,
    )


def main():
    parser = argparse.ArgumentParser(description='Train DR detection models')
    parser.add_argument('--arch', required=True,
                        choices=['cnn', 'cnn_tanh', 'cnn_mcd', 'resnet50_mcd',
                                 'cnn_5class', 'resnet50_5class',
                                 'vgg16', 'vgg_scratch', 'resnet50',
                                 'densenet121', 'xception'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=10,
                        help='EarlyStopping patience (epochs)')
    parser.add_argument('--finetune', action='store_true',
                        help='After head training, unfreeze top layers and continue')
    parser.add_argument('--finetune-epochs', type=int, default=30)
    parser.add_argument('--no-class-weight', action='store_true',
                        help='Disable class weighting (off by default = balanced)')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    inputs_dir = os.path.join(project_root, 'inputs')
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)

    labels = get_labels(os.path.join(inputs_dir, 'labels.csv'))
    images = read_all_images(os.path.join(inputs_dir, 'images'), use_pickle=True)
    is_multiclass = '_5class' in args.arch
    data = build_input_df(images, labels, multiclass=is_multiclass)
    sizes = check_image_sizes(images)
    if not isinstance(sizes, dict):
        raise RuntimeError('Images have inconsistent sizes; cannot continue.')
    image_size = (sizes['width'], sizes['height'])
    class_mode = 'categorical' if is_multiclass else 'binary'
    loss_fn = 'categorical_crossentropy' if is_multiclass else 'binary_crossentropy'

    spec = get_arch_spec(args.arch, inputs_dir, image_size)
    model_path = os.path.join(results_dir, spec.model_filename)
    history_path = os.path.join(results_dir, spec.history_filename)

    train_df, val_df, test_df = stratified_split(
        data, test_size=0.15, val_size=0.15, random_state=args.seed)
    logger.info('Splits — train: %d, val: %d, test: %d',
                len(train_df), len(val_df), len(test_df))

    train_gen = build_train_datagen(
        preprocessing_function=spec.preprocess,
        use_grayscale=spec.use_grayscale,
    ).flow_from_dataframe(
        dataframe=train_df, x_col='filepath', y_col='label',
        target_size=image_size, batch_size=args.batch_size, class_mode=class_mode,
        seed=args.seed,
    )
    val_gen = build_eval_datagen(
        preprocessing_function=spec.preprocess,
        use_grayscale=spec.use_grayscale,
    ).flow_from_dataframe(
        dataframe=val_df, x_col='filepath', y_col='label',
        target_size=image_size, batch_size=args.batch_size, class_mode=class_mode,
        shuffle=False,
    )
    test_gen = build_eval_datagen(
        preprocessing_function=spec.preprocess,
        use_grayscale=spec.use_grayscale,
    ).flow_from_dataframe(
        dataframe=test_df, x_col='filepath', y_col='label',
        target_size=image_size, batch_size=args.batch_size, class_mode=class_mode,
        shuffle=False,
    )

    class_weight = None
    if not args.no_class_weight:
        class_weight = compute_balanced_class_weights(train_df['label'])
        logger.info('Class weights: %s', class_weight)

    model, base_model = spec.builder()
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=loss_fn,
        metrics=['accuracy'],
    )
    model.summary(print_fn=logger.info)

    callbacks = make_callbacks(model_path, patience=args.patience)
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weight,
    )
    history_dict = history.history

    if args.finetune and base_model is not None:
        ft_history = fine_tune(
            model, base_model, train_gen, val_gen,
            epochs=args.finetune_epochs, callbacks=callbacks,
            class_weight=class_weight, loss_fn=loss_fn,
        )
        if ft_history is not None:
            for k, v in ft_history.history.items():
                history_dict.setdefault(k, []).extend(v)

    with open(history_path, 'wb') as f:
        pickle.dump(history_dict, f)

    test_loss, test_acc = model.evaluate(test_gen, verbose=2)
    logger.info('Held-out test loss: %.4f, accuracy: %.4f', test_loss, test_acc)

    plot_nn_loss(history_dict,
                 os.path.join(results_dir, f'{spec.name}_loss.png'),
                 title=f'{spec.name} loss')
    plot_nn_accuracy(history_dict,
                     os.path.join(results_dir, f'{spec.name}_accuracy.png'),
                     title=f'{spec.name} accuracy')


if __name__ == '__main__':
    main()
