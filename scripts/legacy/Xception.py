import os
import pickle
import pandas as pd
from helpers import get_labels, read_all_images, check_image_sizes, build_input_df, build_data_generator
from tensorflow.keras import layers, models, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import Xception
from sklearn.model_selection import train_test_split
from visualization import plot_input_data_histogram, plot_image_from_array, plot_Xception_loss, plot_Xception_accuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
inputs_dir_path = os.path.join(project_root, "inputs")
results_dir_path = os.path.join(project_root, "results")
os.makedirs(results_dir_path, exist_ok=True)
model_path = os.path.join(results_dir_path, 'xception_model.h5')
model_history_path = os.path.join(results_dir_path, 'xception_model_history.pcl')

# Read data
labels_csv_path = os.path.join(inputs_dir_path, 'labels.csv')
labels = get_labels(labels_csv_path)
images_dir_path = os.path.join(inputs_dir_path, 'images')
images = read_all_images(images_dir_path, True)
data = build_input_df(images, labels)

# Plot input data histogram
plot_input_data_histogram(data, os.path.join(results_dir_path, 'input_data_histogram.png'))

# Define some variables
sizes = check_image_sizes(images)
if isinstance(sizes, dict):
    image_width = sizes['width']
    image_height = sizes['height']
else:
    print('Please check the image sizes, because they need to be equal!')

batch_size = 32
epochs = 100

# Split the data into train and test
data_train, data_test = train_test_split(data, test_size=0.2, random_state=123)

# Define data generators
datagen_train = ImageDataGenerator(rescale=1./255)
datagen_test = ImageDataGenerator(rescale=1./255)

generator_train = datagen_train.flow_from_dataframe(
    dataframe=data_train,
    x_col="filepath",
    y_col="label",
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='binary'
)

generator_test = datagen_test.flow_from_dataframe(
    dataframe=data_test,
    x_col="filepath",
    y_col="label",
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Define recreate_model
recreate_model = False

# Xception
if not recreate_model and os.path.isfile(model_path):
    model = load_model(model_path)
    with open(model_history_path, 'rb') as file:
        history = pickle.load(file)
else:
    weights_path = os.path.join(inputs_dir_path, 'xception_weights.h5')
    weights_source = weights_path if os.path.isfile(weights_path) else 'imagenet'
    base_model = Xception(weights=weights_source, include_top=False, input_shape=(image_height, image_width, 3))
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    predictions = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(
        generator_train,
        steps_per_epoch=generator_train.samples // batch_size,
        epochs=epochs,
        validation_data=generator_test,
        validation_steps=generator_test.samples // batch_size
    )
    history = history.history
    
    with open(model_history_path, 'wb') as file:
        pickle.dump(history, file)
    
    model.save(model_path)

print('Evaluating model...')
test_loss, test_acc = model.evaluate(generator_test, verbose=2)
print('\nTest accuracy:', test_acc)

# Confusion Matrix
y_true = data_test['label'].values
y_pred = model.predict(generator_test).flatten()
threshold = 0.5  # Adjust threshold if needed
y_pred_binary = (y_pred > threshold).astype(int)
cm = confusion_matrix(y_true.astype(int), y_pred_binary)

print('Confusion Matrix:')
print(cm)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_true.astype(int), y_pred_binary, pos_label='0')

roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig(os.path.join(results_dir_path, 'roc_curve.png'))

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, y_pred,pos_label='0')
plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.savefig(os.path.join(results_dir_path, 'precision_recall_curve.png'))

loss_plot_path = os.path.join(results_dir_path, 'xception_loss.png')
plot_Xception_loss(history, loss_plot_path)

accuracy_plot_path = os.path.join(results_dir_path, 'xception_accuracy.png')
plot_Xception_accuracy(history, accuracy_plot_path)
