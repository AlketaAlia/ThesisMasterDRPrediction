import os
import pickle
from helpers import get_labels, read_all_images, check_image_sizes, build_input_df, build_data_generator
from visualization import plot_input_data_histogram, plot_VGG16_loss, plot_VGG16_accuracy
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping

# Read the input data
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
inputs_dir_path = os.path.join(project_root, "inputs")
results_dir_path = os.path.join(project_root, "results")
os.makedirs(results_dir_path, exist_ok=True)

images_dir_path = os.path.join(inputs_dir_path, 'images')
labels_csv_path = os.path.join(inputs_dir_path, 'labels.csv')

labels = get_labels(labels_csv_path)
images = read_all_images(images_dir_path, True)
data = build_input_df(images, labels)

plot_input_data_histogram(data, os.path.join(results_dir_path, 'input_data_histogram.png'))

# Define some variables
sizes = check_image_sizes(images)
if isinstance(sizes, dict): # means all the images have the same scale
    image_width = sizes['width']
    image_height = sizes['height']
else:
    print('Please check the image sizes, because they need to be equal!')

batch_size = 32
epochs = 100

model_path = os.path.join(results_dir_path, 'model_vgg16.h5')
model_history_path = os.path.join(results_dir_path, 'model_history_vgg16.pcl')
recreate_model = False

# Build train and test data
data_train, data_test = train_test_split(data, test_size=0.2, random_state=123)

datagen = build_data_generator()

generator_train = datagen.flow_from_dataframe(
    dataframe=data_train,
    x_col="filepath",
    y_col="label",
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='binary'
)

generator_test = datagen.flow_from_dataframe(
    dataframe=data_test,
    x_col="filepath",
    y_col="label",
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Define the VGG16 model
def vgg16_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
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
        layers.Dense(1, activation='sigmoid')
    ])
    return model

if not recreate_model and os.path.isfile(model_path):
    model = load_model(model_path)
    with open(model_history_path, 'rb') as file:
        history = pickle.load(file)
else:
    # Create the VGG16 model
    model = vgg16_model((image_height, image_width, 3))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model with early stopping
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)

    history = model.fit(
        generator_train,
        steps_per_epoch=generator_train.samples // batch_size,
        epochs=epochs,
        validation_data=generator_test,
        validation_steps=generator_test.samples // batch_size,
        callbacks=[early_stopping]
    )
    history = history.history

    # Save the training history
    with open(model_history_path, 'wb') as file:
        pickle.dump(history, file)

    # Save the model
    model.save(model_path)

# Evaluate the model
print('Evaluating model...')
test_loss, test_acc = model.evaluate(generator_test, verbose=2)
print('\nTest accuracy:', test_acc)

# Plot loss
loss_plot_path = os.path.join(results_dir_path, 'nn_loss_vgg16.png')
plot_VGG16_loss(history, loss_plot_path)

# Plot accuracy
accuracy_plot_path = os.path.join(results_dir_path, 'nn_accuracy_vgg16.png')
plot_VGG16_accuracy(history, accuracy_plot_path)
