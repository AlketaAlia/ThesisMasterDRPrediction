import os
import pickle
from helpers import get_labels, read_all_images, check_image_sizes, build_input_df, grayscale_conversion, build_data_generator
from tensorflow.keras import layers, models, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from sklearn.model_selection import train_test_split
from visualization import plot_input_data_histogram, plot_image_from_array, plot_resnet_loss, plot_resnet_accuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
inputs_dir_path = os.path.join(project_root, "inputs")
results_dir_path = os.path.join(project_root, "results")
os.makedirs(results_dir_path, exist_ok=True)
model_path = os.path.join(results_dir_path, 'resnet_model.h5')
model_history_path = os.path.join(results_dir_path, 'resnet_model_history.pcl')

# Read data
labels_csv_path = os.path.join(inputs_dir_path, 'labels.csv')
labels = get_labels(labels_csv_path)
images_dir_path = os.path.join(inputs_dir_path, 'images')
images = read_all_images(images_dir_path, True)
data = build_input_df(images, labels)

# Plot input data histogram
plot_input_data_histogram(data, os.path.join(results_dir_path, 'input_data_histogram.png'))

# Test a grayscale conversion
plot_image_from_array(grayscale_conversion(data.iloc[0]['image']))

# Define some variables
sizes = check_image_sizes(images)
if isinstance(sizes, dict): # means all the images have the same scale
    image_width = sizes['width']
    image_height = sizes['height']
else:
    print('Please check the image sizes, because they need to be equal!')

batch_size = 32
epochs = 100

# Split the data into train and test
data_train, data_test = train_test_split(data, test_size=0.2, random_state=123)

# Define data generators
datagen_train = ImageDataGenerator(rescale=1./255)  # You might need to adjust other parameters
datagen_test = ImageDataGenerator(rescale=1./255)   # Based on your data preprocessing needs

generator_train = datagen_train.flow_from_dataframe(
    dataframe=data_train,
    x_col="filepath",
    y_col="label",
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='binary'
)

generator_test = datagen_test.flow_from_dataframe(
    dataframe=data_test,
    x_col="filepath",
    y_col="label",
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Define recreate_model
recreate_model = False

# ResNet
if not recreate_model and os.path.isfile(model_path): # it means the model exists
    model = load_model(model_path)
    with open(model_history_path, 'rb') as file:
        history = pickle.load(file)
else:
    # Define the ResNet model
    weights_path = os.path.join(inputs_dir_path, 'resnet50_weights.h5')
    weights_source = weights_path if os.path.isfile(weights_path) else 'imagenet'
    base_model = ResNet50(weights=weights_source, include_top=False, input_shape=(image_height, image_width, 3))
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    predictions = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model
    history = model.fit(
        generator_train,
        steps_per_epoch=generator_train.samples // batch_size,
        epochs=epochs,
        validation_data=generator_test,
        validation_steps=generator_test.samples // batch_size
    )
    history = history.history
    # Save the training history to a file
    with open(model_history_path, 'wb') as file:
        pickle.dump(history, file)
    
    # Save the model
    model.save(model_path)

# Evaluate the model
print('Evaluating model...')
test_loss, test_acc = model.evaluate(generator_test, verbose=2)
print('\nTest accuracy:', test_acc)

# Plot loss
loss_plot_path = os.path.join(results_dir_path, 'resnet_loss.png')
plot_resnet_loss(history, loss_plot_path)

# Plot accuracy
accuracy_plot_path = os.path.join(results_dir_path, 'resnet_accuracy.png')
plot_resnet_accuracy(history, accuracy_plot_path)
