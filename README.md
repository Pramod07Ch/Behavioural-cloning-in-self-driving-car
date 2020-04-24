# Self-Driving car Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.
#### Oultine
1. Data collection

2. Data Processing

3. Model architecture

4. Model Training and Testing

5. Conclusion

### Data collection

Data is collected by driving few laps on udacity driving simulator. Because of the nature of the track, so our processing step will also include data augmentation and balancing, in other to prevent our model from being biased towards driving straight and left turns.
Sample images:
[center](images/center.jpg)
![left](images\left.jpg)
![right](images\right.jpg)

### Data Pre-processing

Data processing is done to allow our model to be able to easily work with raw data for training with some augumentation to avialble images.

The pre-processing steps applied are:

1)  The simulator provides three camera views: center, left and right views. Here, we are required to use only one camera view among three views: center, left and right camera images. Here, all images are usedindividaully.While using the left and right images, we add and subtract 0.25 to the steering angles respectively to make up for the camera offsets.
2) simulate different brightness occasions by converting image to HSV channel and randomly scaling the V channel and also add shadow regions to images.
3) In order to balance left and right images, we randomly flip images and change sign on the steering angles.

### Model architecture

The Nvidia model was adopted for training because of its prior application. The network consists of 9 layers, including a normalization layer, 5 convolutional layers and 3 fully connected layers with droupouts inbetween.
Input image mage should be normalized in the first layer and followed by crop of road i.e. only road in the image and removing the unnecessary parts of image such as sky, trees etc.
![nvidia](images\nvidia.png)
Convolution were used in the first three layers with 2x2 strides and a 5x5 kernel, and non-strided convolution with 3x3 kernel size in the last two convolutional layers.

The convolutional layers were followed by three fully connected layers which then outputs the steering angle.

Overfitting was reduced by using aggressive Dropout (0.3) on all the layers. This turned out to be a good practice as the model could generalize to the second track, without using it for training.

An Adam optimizer was used for optimization. This requires little or no tunning as the learning rate is adaptive. In addition, checkpoint and early stop mechanisms can also be used during training to chose best training model by monitoring the validation loss and stopping the training if the loss does not reduce in three consecutive epochs.

### Model train and test
he trained model was tested on the first track. This gave a good result, as the car could drive on the track smoothly.

### Conclusion
A deep learning model to drive autonomously on a simulated track was trained by using a human behaviour-sampled data. The model, which includes 5 layers of convolution and 3 more fully connected layers, was able to learn by cloning human behaviour, and was able to generalize response to a new test track. Though the model only controls the steering angle, it can also be extended to control throttle and brake.

##### Furthur imporvements:
1) Data agumentation as a key part in learning process. Implementation of different techniques make model more generalized.
2) More data would mostly improve the learning efficiency.
3) Use data processing techniques which add value to model generalization i.e. looking into performance of model and select the techniques rather than implementing as a direct technique.

