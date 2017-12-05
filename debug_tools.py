from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from get_set_class_index import getAllCharacterFromIndex


'''
Helper-function for plotting images
Function used to plot 9 images in a 3x3 grid, and writing the true and predicted classes below each image.
'''
def plot_images(images, img_shape,cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 42


    # Create figure with 7x6 sub-plots.
    fig, axes = plt.subplots(7, 6)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        # import cv2
        # cv2.imshow('image',images[0][0])
        # cv2.waitKey()
        ax.imshow(images[i][0], cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:

            #to find corrrect class index
            cls = np.where(cls_true[i] == 1)[0][0]
            xlabel = "True: {0}, Pred: {1}".format(getAllCharacterFromIndex(cls), getAllCharacterFromIndex(cls_pred[i]))

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def plot_example_errors(cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.test.cls[incorrect]

    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


def plot_confusion_matrix(cls_true ,cls_pred, num_classes):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    # cls_true = data.test.cls

    cls = []
    #converting multidimentional array to its correct class
    for classs in cls_true:
        cls.append(np.where(classs == 1)[0][0])

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()



