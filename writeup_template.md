# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[histogram]: ./examples/placeholder_small.png "Histogram of steering angles in data set"


## Rubric Points
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* train.py contains the functions to define model architecture and handle i/o to train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I use the nvidia model as described in the 2016 paper https://arxiv.org/abs/1604.07316

The model consists of a convolution neural network with 3 layers of 5x5 filter sizes and depths 24, 36, and 48, followed by 2 layers of 3x3 filter sizes with depth 64 each.

This is followed by 3 fully connected layers of widths 100, 50, and 10, and a final fully connected layer producing a single output scalar.

The model includes RELU layers to introduce nonlinearity specified as part of the Keras layer, and the data is normalized in the model using a Keras lambda layer (train.py code line 69) to scale the 8 bit image feature to the range -1 to 1.

#### 2. Attempts to reduce overfitting in the model

Dropout is performed on the first two fully connected layers (train.py lines 78, 80) to avoid overfitting.

A validation set was prepared from the total data set using Keras train_test_split, using 20% of the data as the validation set (train.py line 104)

Using the model.fit_generator from Keras, samples were provided using separate training set and validation set generators.

The model keeps the vehicle on track in both forward and reverse directions, indicating good generalization of the model.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (train.py line 105).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving at different speeds, and driving the course in reverse.

For details about how I created the training data, see the next section. 

###  Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a convnet that senses image features which can be mapped to a regression output.

I used the nvidia model because it seems rather successful and incorporates sufficient complexity to handle the features found in lane boundary detection.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I directly applied dropout of 50% on the first two fully connected layers and find that validation mse is close to training mse.

The final step was to run the simulator to see how well the car was driving around track one. It was magic to see the car smoothly following the sweeping left turn and follow the road onto the bridge. The car would not turn sharply enough at the sharp right bend so I incorporated more training data involving carefully driving around the turn slowly.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (train.py lines 68-82) consisted of the nvidia convolution network model with scaling and a cropping layer appended to the front and dropout in the first two fully connected layers at the end. The final fully connected layer generates a single output scalar.


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded several laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer towards the middle of the lane if it veered away from the center. I took a couple of examples throughout the track including examples from the bridge, in turns, and on the straight part of the track. These images show what a recovery looks like:

![alt text][image3]
![alt text][image4]
![alt text][image5]

I did not bother with track 2 because I can hardly drive it without going off track as a human. Without clean training data there is no hope for the model to learn how to drive track 2. I would like to revisit this in a future on a rainy day.

I also drove the track a couple laps in the reverse direction and also included a lap of driving slowly through the turns to increase the number of frames in the tricky turn sections.

To augment the data set, I also flipped images and angles to prevent a bias in the data towards left turns because the track is counterclockwise. This makes the data more balanced as seen by the following histogram: there are equal numbers of negative steering angle frames as positive steering angle frames

![alt text][histogram]

Here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had  number of data points. I left this data in pandas dataframes, which will become input to the generator that provides batches of training data to the model.

I used Keras to split a random 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was around 20 as evidenced by the loss plots. I used an adam optimizer so that manually training the learning rate wasn't necessary.

The generator takes a pandas dataframe, and loops over the data in the set. It first shuffles the data by creating an index to read the dataframe entries by making random choices from the set without replacement so that it will cycle through all of the samples in the set before repeating. Once it has outputted all of the samples in the set, it shuffles again, and loops around.

For each batch, it reads out the image path, and a flag to flip the image if specified. The generator reads the image using cv2, flipping the image as specified.

Because the generator is producing raw image data, preprocessing is thus done inline in keras using the lambda layer to rescale the dynamic range to [-1,1], and crop the image between the hood of the car and the horizon.

At first my model was showing erratic behavior, which I hypothesized was due to bad training examples. I sifted through the capture data and deleted lines from the csv driving log to keep only clean driving examples. This allowed the model to drive around the track more smoothly and turn sharply and smoothly through the tight corners. I'm very impressed with its ability to turn around the sharp corners, which may even exceed my ability to do so on a consistent basis. It is impressive too that I can turn the car around and the model can drive the car smoothly around the corners in reverse and succesfully complete laps around the track.