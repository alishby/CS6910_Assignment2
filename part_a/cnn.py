import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import InputLayer, Conv2D, MaxPooling2D, Dense, Flatten, Activation, BatchNormalization, Dropout
import wandb
from wandb.keras import WandbCallback

class CNN(object):
    def __init__(self, input_shape):
        """
        input_shape: Tuple - Shape of the input layer.
        """
        #Define the model (Stack of Layers)
        self.model = Sequential()
        #Define the input shape
        self.input_shape = input_shape
        #Add an input layer.
        self.model.add(InputLayer(input_shape = self.input_shape))

    def __str__(self):
        #String Representation of CNN class.
        return self.model.summary()
        
    def add_conv_pool_block(self, num_filters = 32, filter_size = (2,2), pool_size= (2,2), activation_fn = "relu", batch_norm = True, dropout = 0.2):
        """
        num_filters : Integer - Number of filters in the Convolution Layer.
        filter_size : Tuple of Integers - Dimensions of the convolution filter kernel.
        pool_size : Tuple of Integers - Dimensions of the pool kernel.
        activation_fn : String -  Activation Function to be used in Convolutional layers.
        batch_norm : Boolean - Indicates whether Batch Normalization is to be used or not.
        Dropout : Integer - Specifies fraction of units to drop.
        """

        #Convloution operation.
        self.model.add(Conv2D(num_filters, filter_size, padding="same", activation = activation_fn))

        #Batch Normalization.
        if batch_norm == True:
            self.model.add(BatchNormalization())

        #Add Max Pooling Layer
        self.model.add(MaxPooling2D(pool_size=pool_size))

        #Dropout : The parameter here specifes fraction of units to drop. [1]
        self.model.add(Dropout(dropout))

    def add_dense_layer(self, dense_neurons, activation_fn = "relu"):
        """
        dense_neurons : Integer - Number of neurons in the dense layer.
        activation_fn : String -  Activation Function to be used in Convolutional layers.
        """
        #Flattens out output after convolutions.
        self.model.add(Flatten())
        #Adds a dense layer.
        self.model.add(Dense(dense_neurons, activation=activation_fn))

    def add_output_layer(self, num_classes):
        """
        num_classes: int - Number of output classes.
        """
        #Add an output layer of 'num_classes'.
        self.model.add(Dense(num_classes, activation='softmax'))

    def build_model(self, num_conv_layers, num_filters, filter_size, pool_size, activation_fn, batch_norm, dropout, dense_neurons, num_classes):
        """
        num_conv_layers : Integer - Number of Convolution Layers in the network.
        num_filters : List of Integers - Number of filters corresponding to each convolutional layer.
        filter_size : Tuple of Integers - Dimensions of the filter kernel corresponding to each convolutional layer.
        pool_size : Tuple of Integers - Dimensions of the pool kernel corresponding to each pool layer.
        activation_fn : String -  Activation Function to be used in Convolutional layers.
        batch_norm : Boolean - Indicates whether Batch Normalization is to be used or not.
        Dropout : Integer - Specifies fraction of units to drop.
        dense_neurons : Integer - Number of neurons in the dense layer.
        num_classes: int - Number of output classes.
        """
        #Add convolution-pool blocks
        for i in range(0, num_conv_layers):
            self.add_conv_pool_block(num_filters = num_filters[i], filter_size=filter_size[i], pool_size=pool_size[i], activation_fn=activation_fn, batch_norm=batch_norm, dropout=dropout)
        self.add_dense_layer(dense_neurons=dense_neurons)
        #Add output layer
        self.add_output_layer(num_classes = num_classes)

    def train(self, train_data, val_data, optimizer, learning_rate, loss_fn, num_epochs, batch_size):
        """
        train_data : TF Data Generator for train data.
        val_data : TF Data Generator for validation data.
        optimizer : string corresponding to the name of the optimizer to invoke.
        learning_rate : Integer -  Learning Rate to train the model.
        loss_fn : Loss Function.
        num_epochs : Integer - Number of epochs to train for.
        batch_size : Integer - Batch size of train and val generator.
        """
        #Define the optimizer
        opt = eval('keras.optimizers.' + optimizer + '(learning_rate = learning_rate)')
        #Compile the model.
        self.model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])
        #Fit the model on train data.
        self.model.fit(train_data, validation_data=val_data, epochs = num_epochs, batch_size=batch_size, callbacks=[WandbCallback()])

    def test(self, test_data):
        """
        test_data : Data Generator for test data.
        """
        self.model.evaluate(test_data)

#[1] https://machinelearningmastery.com/how-to-reduce-overfitting-with-dropout-regularization-in-keras/