import argparse
from preprocess import generate_batch_train_val
from preprocess import generate_batch_test
from cnn import *
import tensorflow as tf 

def filter_list(s):
    try:
        x, y = map(int, s.split(','))
        return x, y
    except:
        raise argparse.ArgumentTypeError("Filters must have size x,y")

def pool_list(s):
    try:
        x, y = map(int, s.split(','))
        return x, y
    except:
        raise argparse.ArgumentTypeError("Pooling must have size x,y")


#Define the Command Line Arguments
parser = argparse.ArgumentParser(description='Set the directory paths, hyperparameters of the model.')
parser.add_argument('--augmentation', action='store_true')
parser.add_argument('--train_path', type=str, default='inaturalist_12K/train/', help='Path of the train data directory.')
parser.add_argument('--test_path', type=str, default='inaturalist_12K/val/', help='Path of the test data directory')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--image_size', type=int, nargs='+', help='Image size, 2d, (height, width)', required=True)
parser.add_argument('--num_conv_layers', type=int, default=5, help='Number of Convolution Pool Blocks')
parser.add_argument('--num_epochs', type=int, default=5, help='Number of Epochs')
parser.add_argument('--num_filters', type=int, nargs='+', help='Number of Filters in Convolution Layer', required=True)
parser.add_argument('--filter_size', type=filter_list, nargs=5, help='Filter size in each convolution layer, comma seperated', required=True)
parser.add_argument('--pool_size', type=pool_list, nargs=5, help='Pool size in each MaxPool layer, comma seperated', required=True)
parser.add_argument('--dense_neurons', type=int, help='Neurons in Dense Layer after Convoltions', default=128)
parser.add_argument('--batch_norm', action='store_true')
parser.add_argument('--dropout', type=float, default=0, help='Dropout Rate')


#Parse the arguments
args = parser.parse_args()
augmentation = args.augmentation
train_path = args.train_path
test_path = args.test_path
batch_size = args.batch_size
learning_rate = args.learning_rate
image_size = args.image_size
num_conv_layers = args.num_conv_layers
num_filters = args.num_filters
filter_size = list(args.filter_size)
pool_size = list(args.pool_size)
dense_neurons = args.dense_neurons
batch_norm = args.batch_norm
dropout = args.dropout
num_epochs = args.num_epochs


#Affirm that dimension are appropriate.
if len(image_size) != 2:
    raise ValueError("Image size is expected to be 2 dimensional.")

if num_conv_layers != len(num_filters) or num_conv_layers !=len(filter_size):
    raise ValueError("Dimension Mismatch")

#Generate training, validation and test batches.
train_data, val_data, class_labels = generate_batch_train_val(train_path, augmentation = augmentation, batch_size = batch_size, image_size = tuple(image_size))
test_data = generate_batch_test(test_path, batch_size = batch_size, image_size = tuple(image_size))

print(class_labels)

#As, our input has 3 channels RGB, our input_shape would have (height, width, channels = 3)
input_shape= image_size
input_shape.append(3)
cnn = CNN(tuple(input_shape))

#Builds the CNN model - Defines the layers as per the command-line arguments.
cnn.build_model(num_conv_layers, num_filters, filter_size, pool_size, activation_fn = "relu", batch_norm = batch_norm, dropout = dropout, dense_neurons = dense_neurons, num_classes = len(class_labels))
#Displays the Model summary.
print(cnn)
#Trains the model as per the configuration provided in the command-line arguments.
cnn.train(train_data, val_data, optimizer = "Adam", learning_rate = learning_rate, loss_fn = 'categorical_crossentropy', num_epochs = num_epochs, batch_size = batch_size)
#Evaluates the performance of the model on the test set.
cnn.test(test_data)