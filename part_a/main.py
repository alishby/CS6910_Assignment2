import argparse
from preprocess import generate_batch_train_val
from preprocess import generate_batch_test

#Define the Command Line Arguments
parser = argparse.ArgumentParser(description='Set the directory paths, hyperparameters of the model.')
parser.add_argument('--augmentation', action='store_true')
parser.add_argument('--train_path', type=str, default='inaturalist_12K/train/', help='Path of the train data directory.')
parser.add_argument('--test_path', type=str, default='inaturalist_12K/val/', help='Path of the test data directory')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--image_size', type=int, nargs='+', help='Image size, 2d', required=True)

#Parse the arguments
args = parser.parse_args()
augmentation = args.augmentation
train_path = args.train_path
test_path = args.test_path
batch_size = args.batch_size
image_size = args.image_size

#Affirm that dimension of image size is appropriate.
if len(image_size) != 2:
    raise ValueError("Image size is expected to be 2 dimensional.")

#Generate training, validation and test batches.
train_data, val_data, class_labels = generate_batch_train_val(train_path, augmentation = augmentation, batch_size = batch_size, image_size = tuple(image_size))
test_data = generate_batch_test(test_path, batch_size = batch_size, image_size = tuple(image_size))

print(class_labels)