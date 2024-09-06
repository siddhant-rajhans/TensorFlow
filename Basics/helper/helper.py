# Import necessary libraries
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import datetime
import zipfile
import os

# Define a function to load and preprocess an image
def load_and_prep_image(filename, img_shape=224, scale=True):
    """
    Loads an image from a file and preprocesses it for use with a model.

    Args:
        filename (str): Path to the image file.
        img_shape (int): Size to resize the image to. Defaults to 224.
        scale (bool): Whether to scale pixel values to the range (0, 1). Defaults to True.

    Returns:
        Preprocessed image tensor.
    """
    # Read in the image
    img = tf.io.read_file(filename)
    # Decode the image into a tensor
    img = tf.image.decode_jpeg(img)
    # Resize the image
    img = tf.image.resize(img, [img_shape, img_shape])
    # Scale the image if necessary
    if scale:
        return img / 255.
    else:
        return img

# Define a function to create a confusion matrix
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False):
    """
    Creates a labelled confusion matrix comparing predictions and ground truth labels.

    Args:
        y_true: Array of truth labels.
        y_pred: Array of predicted labels.
        classes: Array of class labels. If None, integer labels are used.
        figsize: Size of output figure. Defaults to (10, 10).
        text_size: Size of output figure text. Defaults to 15.
        norm: Normalize values or not. Defaults to False.
        savefig: Save confusion matrix to file or not. Defaults to False.

    Returns:
        A labelled confusion matrix plot comparing y_true and y_pred.
    """
    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Normalize the confusion matrix if necessary
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    # Get the number of classes
    n_classes = cm.shape[0]

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    # Create the heatmap
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    # Add a colorbar
    fig.colorbar(cax)

    # Get the class labels
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Set the title and labels
    ax.set(title="Confusion Matrix",
           xlabel="Predicted label",
           ylabel="True label",
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=labels,
           yticklabels=labels)

    # Move the x-axis labels to the bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)

    # Save the figure to file if necessary
    if savefig:
        fig.savefig("confusion_matrix.png")

# Define a function to predict and plot an image
def pred_and_plot(model, filename, class_names):
    """
    Imports an image, makes a prediction on it with a trained model, and plots the image with the predicted class as the title.

    Args:
        model: Trained model.
        filename: Path to the image file.
        class_names: Array of class names.
    """
    # Load and preprocess the image
    img = load_and_prep_image(filename)
    # Make a prediction on the image
    pred = model.predict(tf.expand_dims(img, axis=0))

    # Get the predicted class
    if len(pred[0]) > 1:
        pred_class = class_names[pred.argmax()]
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])]

    # Plot the image and predicted class
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)

# Define a function to create a TensorBoard callback
def create_tensorboard_callback(dir_name, experiment_name):
    """
    Creates a TensorBoard callback instance to store log files.

    Args:
        dir_name: Target directory to store TensorBoard log files.
        experiment_name: Name of experiment directory.

    Returns:
        TensorBoard callback instance.
    """
    # Create the log directory
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Create the TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    # Print the log directory
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback

# Define a function to plot loss curves
def plot_loss_curves(history):
    """
    Plots separate loss curves for training and validation metrics.

    Args:
        history: TensorFlow model History object.
    """
    # Get the loss and accuracy values
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    # Get the number of epochs
    epochs = range(len(history.history['loss']))

    # Plot the loss curves
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot the accuracy curves
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

# Define a function to compare histories
def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compares two TensorFlow model History objects.

    Args:
        original_history: History object from original model.
        new_history: History object from continued model training.
        initial_epochs: Number of epochs in original_history.
    """
    # Get the accuracy and loss values
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine the original history with the new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Create the figure and axis
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    # Plot the accuracy curves
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    # Plot a line to indicate the start of fine-tuning
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    # Plot the loss curves
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    # Plot a line to indicate the start of fine-tuning
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

# Define a function to unzip a zip file
def unzip_data(filename):
    """
    Unzips a zip file into the current working directory.

    Args:
        filename (str): Path to the zip file.
    """
    # Open the zip file
    zip_ref = zipfile.ZipFile(filename, "r")
    # Extract the contents of the zip file
    zip_ref.extractall()
    # Close the zip file
    zip_ref.close()

# Define a function to walk through a directory
def walk_through_dir(dir_path):
    """
    Walks through a directory and prints the number of subdirectories and files.

    Args:
        dir_path (str): Path to the directory.
    """
    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(dir_path):
        # Print the number of subdirectories and files
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

# Define a function to calculate model metrics
def calculate_results(y_true, y_pred):
    """
    Calculates model accuracy, precision, recall, and F1 score.

    Args:
        y_true: Array of true labels.
        y_pred: Array of predicted labels.

    Returns:
        A dictionary of model metrics.
    """
    # Calculate model accuracy
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    # Calculate model precision, recall, and F1 score
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    # Create a dictionary of model metrics
    model_results = {"accuracy": model_accuracy,
                     "precision": model_precision,
                     "recall": model_recall,
                     "f1": model_f1}
    return model_results
   