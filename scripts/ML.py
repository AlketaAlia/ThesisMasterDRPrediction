"""Train Decision Tree, Random Forest, and SVM on CNN-extracted features.

The original implementation flattened raw 224x224x3 pixels into 150,528-dim
feature vectors and trained on those — which scales poorly and offers no
spatial inductive bias. Here we extract 1024-dim features from a frozen
DenseNet121 (ImageNet pretrained) and train classical ML models on top.

Splits are stratified, evaluated on a held-out test set, and class imbalance
is handled with `class_weight='balanced'`.
"""
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, classification_report,
    confusion_matrix, log_loss, precision_recall_curve, roc_curve,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from helpers import (
    build_input_df, check_image_sizes, get_labels, read_all_images,
    stratified_split,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('ml')


def extract_features(df, image_size, feature_extractor, batch_size=32):
    """Run a frozen CNN over a dataframe and return (features, labels)."""
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    gen = datagen.flow_from_dataframe(
        dataframe=df, x_col='filepath', y_col='label',
        target_size=image_size, batch_size=batch_size,
        class_mode='binary', shuffle=False,
    )
    n = gen.samples
    steps = int(np.ceil(n / batch_size))
    features = feature_extractor.predict(gen, steps=steps, verbose=1)
    labels = df['label'].astype(int).values[:n]
    return features[:n], labels


def train_and_evaluate(name, classifier, X_train, y_train, X_test, y_test,
                       results_dir):
    """Fit, evaluate, and save plots for a classical classifier."""
    logger.info('Training %s on %d samples (%d features)',
                name, X_train.shape[0], X_train.shape[1])
    classifier.fit(X_train, y_train)

    y_pred_train = classifier.predict(X_train)
    y_pred_test = classifier.predict(X_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    logger.info('%s — train acc: %.4f, test acc: %.4f',
                name, acc_train, acc_test)

    report = classification_report(y_test, y_pred_test, digits=4)
    logger.info('%s classification report:\n%s', name, report)

    has_proba = hasattr(classifier, 'predict_proba')
    y_prob_test = classifier.predict_proba(X_test) if has_proba else None
    if y_prob_test is not None:
        loss = log_loss(y_test, y_prob_test)
        logger.info('%s — test log loss: %.4f', name, loss)

    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No DR', 'DR'], yticklabels=['No DR', 'DR'])
    plt.title(f'{name} — Test Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{name}_TestConfusionMatrix.png'))
    plt.close()

    if y_prob_test is not None:
        positive_scores = y_prob_test[:, 1]
        fpr, tpr, _ = roc_curve(y_test, positive_scores)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{name} — ROC Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{name}ROCCurve.png'))
        plt.close()

        precision, recall, _ = precision_recall_curve(y_test, positive_scores)
        ap = average_precision_score(y_test, positive_scores)
        plt.figure()
        plt.plot(recall, precision, lw=2, label=f'PR (AP = {ap:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{name} — Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{name}PrecisionRecall.png'))
        plt.close()

    return {
        'name': name,
        'train_acc': acc_train,
        'test_acc': acc_test,
        'report': report,
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    inputs_dir = os.path.join(project_root, 'inputs')
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)

    labels = get_labels(os.path.join(inputs_dir, 'labels.csv'))
    images = read_all_images(os.path.join(inputs_dir, 'images'), use_pickle=True)
    data = build_input_df(images, labels)
    sizes = check_image_sizes(images)
    if not isinstance(sizes, dict):
        raise RuntimeError('Images have inconsistent sizes.')
    image_size = (sizes['width'], sizes['height'])

    train_df, _, test_df = stratified_split(
        data, test_size=0.2, val_size=0.0, random_state=123)
    logger.info('Splits — train: %d, test: %d', len(train_df), len(test_df))

    weights_path = os.path.join(inputs_dir, 'densenet_weights.h5')
    weights_source = weights_path if os.path.isfile(weights_path) else 'imagenet'
    base = DenseNet121(weights=weights_source, include_top=False,
                       input_shape=(*image_size, 3), pooling='avg')
    base.trainable = False
    logger.info('Extracting features with DenseNet121 (frozen)…')

    X_train, y_train = extract_features(train_df, image_size, base)
    X_test, y_test = extract_features(test_df, image_size, base)
    logger.info('Feature shapes — train: %s, test: %s',
                X_train.shape, X_test.shape)

    classifiers = [
        ('DT',
         DecisionTreeClassifier(random_state=123, class_weight='balanced')),
        ('RandomForest',
         RandomForestClassifier(n_estimators=300, random_state=123,
                                class_weight='balanced', n_jobs=-1)),
        ('SVM',
         SVC(kernel='rbf', probability=True, random_state=123,
             class_weight='balanced')),
    ]

    summary = []
    for name, clf in classifiers:
        summary.append(train_and_evaluate(
            name, clf, X_train, y_train, X_test, y_test, results_dir))

    logger.info('Summary:')
    for row in summary:
        logger.info('  %s — train: %.4f, test: %.4f',
                    row['name'], row['train_acc'], row['test_acc'])


if __name__ == '__main__':
    main()
