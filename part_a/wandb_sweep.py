from preprocess import generate_batch_train_val
from preprocess import generate_batch_test
from cnn import *
import tensorflow as tf 
import wandb

def train_wandb(config = None):

    run = wandb.init(config=config, resume=True)
    config = wandb.config

    name = f'num_fil_{config.num_filters}_filter_shape_{config.filter_size}_dense_neurons_{config.dense_neurons}_lr_{config.learning_rate}_aug_{config.augmentation}_epochs_{config.num_epochs}_bnorm_{config.batch_norm}_dropout_{config.dropout}'
    wandb.run.name = name
    wandb.run.save()

    train_path = 'inaturalist_12K/train/' 
    batch_size = 16
    image_size = [224, 224]
    num_conv_layers = 5

    augmentation = config.augmentation
    learning_rate = config.learning_rate
    num_filters = config.num_filters
    filter_size = config.filter_size
    pool_size = config.pool_size
    dense_neurons = config.dense_neurons
    batch_norm = config.batch_norm
    dropout = config.dropout
    num_epochs = config.num_epochs

    #Generate training, validation and test batches.
    train_data, val_data, class_labels = generate_batch_train_val(train_path, augmentation = augmentation, batch_size = batch_size, image_size = tuple(image_size))

    input_shape= image_size
    input_shape.append(3) # 3 channels
    cnn = CNN(tuple(input_shape))

    cnn.build_model(num_conv_layers, num_filters, filter_size, pool_size, activation_fn = "relu", batch_norm = batch_norm, dropout = dropout, dense_neurons = dense_neurons, num_classes = len(class_labels))
    cnn.train(train_data, val_data, optimizer = "Adam", learning_rate = learning_rate, loss_fn = 'categorical_crossentropy', num_epochs = num_epochs, batch_size = batch_size)

project_name = '' #Add project name here
entity = '' #Add username here
wandb.init(project=project_name, entity=entity)

sweep_config = {
    'method': 'bayes', 
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'   
              },
    'early_terminate' : {
      'type': 'hyperband',
      'min_iter': 10
    },
    'parameters': {
        'num_epochs': {
            'values': [20,30]
        },
        'num_filters': {
            'values': [
                [32,64,128,256,512],
                [64,64,64,64,64],
                [512,256,128,64,32]
            ]
        },
        'filter_size' : {
            'values' : [
                [(7,7),(7,7),(5,5),(3,3),(2,2)],
                [(2,2),(3,3),(5,5),(7,7),(7,7)],
                [(3,3),(3,3),(3,3),(3,3),(3,3)],
            ]
        },
        'pool_size' : {
            'values' : [
                [(2,2),(2,2),(2,2),(2,2),(2,2)]
            ]
        },
        'learning_rate': {
            'values': [1e-3,1e-4]
        },
        'batch_norm': {
            'values': [True, False]
        },
        'dense_neurons': {
            'values': [128, 256, 512]
        },
        'augmentation': {
            'values': [True, False]
        },
        'dropout': {
            'values': [0,0.2,0.5]
        }
    }
}

#To add a new agent to an existing sweep, comment next line and directly put sweep_id in wandb.agent
sweep_id = wandb.sweep(sweep_config, project=project_name, entity=entity)

wandb.agent(sweep_id, project=project_name, function=train_wandb)