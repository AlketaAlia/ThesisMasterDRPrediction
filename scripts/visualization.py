import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.style import context


def plot_input_data_histogram(data, save_path = None):
    with context('default'):
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
        label_counts = data['label'].value_counts()
        # ax.hist(data['label'])
        ax.bar(label_counts.index, label_counts.values)
        ax.set_title("Input data histogram")
        ax.set_xlabel("Diagnosis")
        ax.set_ylabel("Occurrence")
        # ax.grid(visible = True, which = 'both')
        plt.show()
        if save_path is not None:
            fig.savefig(save_path)
    
def plot_image_from_array(data, save_path = None):
    with context('default'):
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(data, cmap = 'gray')
        plt.show()
        if save_path is not None:
            fig.savefig(save_path)
            
        
def plot_nn_loss(data, save_path = None, title = "Neural network loss"):
    loss_train = data['loss']
    loss_test = data['val_loss']
    epochs = np.array(range(len(loss_train))) + 1
    with context('default'):
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
        ax.plot(epochs, loss_train, label = 'train loss')
        ax.plot(epochs, loss_test, label = 'validation loss')
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(visible = True, which = 'both')
        ax.legend()
        plt.show()
        if save_path is not None:
            fig.savefig(save_path)
    
def plot_nn_accuracy(data, save_path = None, title = "Neural network accuracy"):
    accuracy_train = data['accuracy']
    accuracy_test = data['val_accuracy']
    epochs = np.array(range(len(accuracy_train))) + 1
    with context('default'):
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
        ax.plot(epochs, accuracy_train, label = 'train accuracy')
        ax.plot(epochs, accuracy_test, label = 'validation accuracy')
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.grid(visible = True, which = 'both')
        ax.legend()
        plt.show()
        if save_path is not None:
            fig.savefig(save_path)


def plot_resnet_loss(data, save_path = None):
    plot_nn_loss(data, save_path, title="ResNet loss")


def plot_resnet_accuracy(data, save_path = None):
    plot_nn_accuracy(data, save_path, title="ResNet accuracy")


def plot_Xception_loss(data, save_path = None):
    plot_nn_loss(data, save_path, title="Xception loss")


def plot_Xception_accuracy(data, save_path = None):
    plot_nn_accuracy(data, save_path, title="Xception accuracy")


def plot_VGG16_loss(data, save_path = None):
    plot_nn_loss(data, save_path, title="VGG16 loss")


def plot_VGG16_accuracy(data, save_path = None):
    plot_nn_accuracy(data, save_path, title="VGG16 accuracy")
