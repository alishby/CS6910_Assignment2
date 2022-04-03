# CS6910_Assignment2 : 

## Learn how to use CNNs: train from scratch, finetune a pretrained model, use a pre-trained model as it is.

The target task to be performed was Image Classification.

The [Inaturalist 12K Dataset](https://storage.googleapis.com/wandb_datasets/nature_12K.zip) having 10 classes was used.

Train data (10K Images) was split into (90% Train -10% val). 

This implementation is based on Tensorflow.

## [Wandb Report](https://wandb.ai/cs6910_a2/CS6910_A2/reports/Assignment-2-Convolutional-Neural-Networks--VmlldzoxNzYwNTgw)

## Part A:- Train from scratch
[Part A](https://github.com/PranjalChitale/CS6910_Assignment2/tree/main/part_a)

In part A, we train a CNN model from scratch.

[Part A readme](https://github.com/PranjalChitale/CS6910_Assignment2/blob/main/part_a/README.md) describes how to use the code to define a CNN model and train it from scratch.

## Part B:- Finetune a pretrained model
In part B, we fine-tune models pretrained on Imagenet on the Inaturalist Dataset.
Particularly, we experimented with InceptionV3, Xception ResNet50, InceptionResNetV2 MobileNetV2.

[Part B readme](https://github.com/PranjalChitale/CS6910_Assignment2/tree/main/part_b) describes how to use the code to fine-tune pretrained models on the Inaturalist Dataset.

## Part C:- Using a Pretrained Model for some application
Application chosen: Pothole Detection using Yolo V4.
We have used the pre-trained weights of YoloV4 by Alexey Bochkovskiy et al. 
In part C, we fine-tune [YoloV4](https://github.com/AlexeyAB/darknet) to detect road potholes in videos.

[Link to the Demo Video](https://drive.google.com/file/d/1K6L8b5KoGcIns-FJYyFSle6RWBV8RjwA/view?usp=sharing)
