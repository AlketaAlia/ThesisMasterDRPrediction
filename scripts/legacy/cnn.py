import os, pickle
from helpers import get_labels, read_all_images, check_image_sizes, build_input_df, grayscale_conversion, build_data_generator
from visualization import plot_input_data_histogram, plot_image_from_array, plot_nn_loss, plot_nn_accuracy
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50, Xception, DenseNet121, VGG16

#%% Read the inpud data
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

# Test a grayscale conversion
plot_image_from_array(grayscale_conversion(data.iloc[0]['image']))

#%% Define some variables
sizes = check_image_sizes(images)
if isinstance(sizes, dict): # means all the images have the same scale
    image_width = sizes['width']
    image_height = sizes['height']
else:
    print('Please check the image sizes, because they need to be equal!')

batch_size = 32
epochs = 100

model_path = os.path.join(results_dir_path, 'vgg_model.h5')
model_history_path = os.path.join(results_dir_path, 'vgg_model_history.pcl')
recreate_model = False

#%% Build train and test data

# Split the data into train and test
data_train, data_test = train_test_split(data, test_size = 0.2, random_state = 123)

datagen = build_data_generator()

generator_train = datagen.flow_from_dataframe(
    dataframe = data_train,
    x_col = "filepath",
    y_col = "label",
    target_size = (image_width, image_height),
    batch_size = batch_size,
    class_mode = 'binary'
)

generator_test = datagen.flow_from_dataframe(
    dataframe = data_test,
    x_col = "filepath",
    y_col = "label",
    target_size = (image_width, image_height),
    batch_size = batch_size,
    class_mode = 'binary'
)

#%% NN
if not recreate_model and os.path.isfile(model_path): # it means the model exists
    model = load_model(model_path)
    with open(model_history_path, 'rb') as file:
        history = pickle.load(file)
else:    
    # Define the CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (image_height, image_width, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation = 'relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation = 'relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation = 'relu'),
        layers.Dense(1, activation = 'sigmoid')
    ])
    
    # # Define the CNN model with a combination of activation functions
    # model = models.Sequential([
    # layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    # layers.MaxPooling2D((2, 2)),
    # layers.Conv2D(64, (3, 3), activation='tanh'),
    # layers.MaxPooling2D((2, 2)),
    # layers.Conv2D(128, (3, 3), activation='relu'),
    # layers.MaxPooling2D((2, 2)),
    # layers.Flatten(),
    # layers.Dense(128, activation='tanh'),
    # layers.Dense(1, activation='sigmoid')
    # ])
    
    # weights_path = os.path.join(inputs_dir_path, 'vgg_weights.h5')
    # base_model = VGG16(weights = weights_path, include_top = False, input_shape = (image_height, image_width, 3))
    # for layer in base_model.layers:
    #     layer.trainable = False
    # x = base_model.output
    # x = layers.GlobalAveragePooling2D()(x)
    # x = layers.Dense(128, activation='relu')(x)
    # predictions = layers.Dense(1, activation='sigmoid')(x)
    # model = models.Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(optimizer = 'adam',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    
    # Train the model
    history = model.fit(
        generator_train,
        steps_per_epoch = generator_train.samples // batch_size,
        epochs = epochs,
        validation_data = generator_test,
        validation_steps = generator_test.samples // batch_size
    )
    history = history.history
    # Save the training history to a file
    with open(model_history_path, 'wb') as file:
        pickle.dump(history, file)
    
    # Save the model
    model.save(model_path)

# Evaluate the model
# print('Evaluating model...')
# test_loss, test_acc = model.evaluate(generator_test, verbose = 2)
# print('\nTest accuracy:', test_acc)

#%% Plot some graphs

# Plot loss
loss_plot_path = os.path.join(results_dir_path, 'vgg_loss.png')
plot_nn_loss(history, loss_plot_path, title = 'VGG loss')

# Plot accuracy
accuracy_plot_path = os.path.join(results_dir_path, 'vgg_accuracy.png')
plot_nn_accuracy(history, accuracy_plot_path, title = 'VGG accuracy')



