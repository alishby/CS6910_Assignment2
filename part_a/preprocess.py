import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def generate_batch_train_val(path, augmentation, batch_size, image_size):
    #Splits the dataset into train and validation.
    #Keras' ImageDataGenerator is used to split data into train and test.
    """
    path : String - Path to the train data.
    augmentation : Boolean - Indicates whether Data Augmentation is to be performed or not.
    batch_size : Integer - Specifies the batch size for train and val data.
    image_size : Tuple of Integers - Specifies the shape to which the images need to be resized.
    """ 
    if augmentation == True:
        #Applies data augmentation if specified
        train_data_gen = ImageDataGenerator(
                            featurewise_center = True,
                            featurewise_std_normalization = True,
                            rescale = 1./255,
                            horizontal_flip = True,
                            rotation_range = 30,
                            shear_range = 0.2,
                            zoom_range = [0.75, 1.75],
                            width_shift_range = 0.2,
                            height_shift_range = 0.2,
                            validation_split = 0.10
                        )
    else:
        train_data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.10)
 
    #Flow from directory expects that images belonging to each class is present in its own folder but inside the same parent folder : data directory.
    #It takes path to the data directory as input and generates batches of desired batch size.
    #Need to specify appropriate subset (training / validation) to generate batches for respective subset.
    
    train_data = train_data_gen.flow_from_directory(
            path,
            target_size=image_size,
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
            seed = 0,
            subset="training"
        )
        
    val_data = train_data_gen.flow_from_directory(
        path,
        target_size=image_size,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed = 0,
        subset="validation"
    )

    #Gets the list of class labels.
    class_labels = list(train_data.class_indices.keys())


    return train_data, val_data, class_labels

def generate_batch_test(path, batch_size, image_size):
    #Generates batches of test data.
    """
    path : String - Path to the train data.
    batch_size : Integer - Specifies the batch size for train and val data.
    image_size : Tuple of Integers - Specifies the shape to which the images need to be resized.
    """
    test_data_gen = ImageDataGenerator(
    featurewise_center = True,
    featurewise_std_normalization = True,
    rescale = 1./255
    )

    test_data = test_data_gen.flow_from_directory(
            path,
            target_size=image_size,
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
            seed=0,
        )

    return test_data

#Reference : Tensorflow Documentation.
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
