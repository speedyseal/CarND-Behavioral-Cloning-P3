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

[center]: ./examples/center_2017_05_03_06_17_10_291.jpg "Center of lane driving"
[recovery1]: ./examples/center_2017_05_03_06_24_04_891.jpg "Recovery Image"
[recovery2]: ./examples/center_2017_05_03_06_24_05_915.jpg "Recovery Image"
[recovery3]: ./examples/center_2017_05_03_06_24_06_423.jpg "Recovery Image"
[image]: ./examples/image.png "Normal Image"
[flippedimage]: ./examples/flippedimage.png "Flipped Image"
[cropped]: ./examples/cropped.png "Cropped Output"
[histogram]: ./examples/histogram.png "Histogram of steering angles in data set"


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


```
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_2 (Lambda)                (None, 160, 320, 3)   0           lambda_input_2[0][0]             
____________________________________________________________________________________________________
cropping2d_2 (Cropping2D)        (None, 90, 320, 3)    0           lambda_2[0][0]                   
____________________________________________________________________________________________________
convolution2d_6 (Convolution2D)  (None, 43, 158, 24)   1824        cropping2d_2[0][0]               
____________________________________________________________________________________________________
convolution2d_7 (Convolution2D)  (None, 20, 77, 36)    21636       convolution2d_6[0][0]            
____________________________________________________________________________________________________
convolution2d_8 (Convolution2D)  (None, 8, 37, 48)     43248       convolution2d_7[0][0]            
____________________________________________________________________________________________________
convolution2d_9 (Convolution2D)  (None, 6, 35, 64)     27712       convolution2d_8[0][0]            
____________________________________________________________________________________________________
convolution2d_10 (Convolution2D) (None, 4, 33, 64)     36928       convolution2d_9[0][0]            
____________________________________________________________________________________________________
flatten_2 (Flatten)              (None, 8448)          0           convolution2d_10[0][0]           
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 100)           844900      flatten_2[0][0]                  
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 100)           0           dense_5[0][0]                    
____________________________________________________________________________________________________
dense_6 (Dense)                  (None, 50)            5050        dropout_3[0][0]                  
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 50)            0           dense_6[0][0]                    
____________________________________________________________________________________________________
dense_7 (Dense)                  (None, 10)            510         dropout_4[0][0]                  
____________________________________________________________________________________________________
dense_8 (Dense)                  (None, 1)             11          dense_7[0][0]                    
====================================================================================================
Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded several laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer towards the middle of the lane if it veered away from the center. I took a couple of examples throughout the track including examples from the bridge, in turns, and on the straight part of the track.

![alt text][recovery1]

![alt text][recovery2]

![alt text][recovery3]


I did not bother with track 2 because I can hardly drive it without going off track as a human. Without clean training data there is no hope for the model to learn how to drive track 2. I would like to revisit this in a future on a rainy day.

I also drove the track a couple laps in the reverse direction and also included a lap of driving slowly through the turns to increase the number of frames in the tricky turn sections.

To augment the data set, I also flipped images and angles to prevent a bias in the data towards left turns because the track is counterclockwise. This makes the data more balanced as seen by the following histogram: there are equal numbers of negative steering angle frames as positive steering angle frames

![alt text][histogram]

Here is an image that has then been flipped:

![alt text][image]
![alt text][flippedimage]

After the collection process, I had  number of data points. I left this data in pandas dataframes, which will become input to the generator that provides batches of training data to the model.

I used Keras to split a random 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was around 20 as evidenced by the loss plots. I used an adam optimizer so that manually training the learning rate wasn't necessary.

The generator takes a pandas dataframe, and loops over the data in the set. It first shuffles the data by creating an index to read the dataframe entries by making random choices from the set without replacement so that it will cycle through all of the samples in the set before repeating. Once it has outputted all of the samples in the set, it shuffles again, and loops around.

For each batch, it reads out the image path, and a flag to flip the image if specified. The generator reads the image using cv2, flipping the image as specified.

Because the generator is producing raw image data, preprocessing is thus done inline in keras using the lambda layer to rescale the dynamic range to [-1,1], and crop the image between the hood of the car and the horizon.

Here is an example cropped image from the model.
![alt text][cropped]

At first my model was showing erratic behavior, which I hypothesized was due to bad training examples. I sifted through the capture data and deleted lines from the csv driving log to keep only clean driving examples. This allowed the model to drive around the track more smoothly and turn sharply and smoothly through the tight corners. I'm very impressed with its ability to turn around the sharp corners, which may even exceed my ability to do so on a consistent basis. It is impressive too that I can turn the car around and the model can drive the car smoothly around the corners in reverse and succesfully complete laps around the track.

#### 4. Analysis of driving videos and future work
The captures of the car camera while driving autonomously have been concatenated into a video and posted to YouTube. The links are shown here.

Forward laps around track 1
[![Video forward](http://img.youtube.com/vi/OBcutxptyAw/0.jpg)](https://youtu.be/OBcutxptyAw)

Reverse laps around track 1
[![Video reverse](http://img.youtube.com/vi/pBrzijOFV7Y/0.jpg)](https://youtu.be/pBrzijOFV7Y)

While the driving could be improved, these examples are interesting because the recovery behavior of the controller is evident. As the car moves towards the edge of the road, the controller takes responsive action to steer the car back towards the center, albeit a bit too enthusiastically. The car stays on the road however, even at 30 mph. Choosing a lower speed shows more smooth driving behavior.

The model in this project is a bit oversimplified. In reality steering angle needs to be speed dependent, creating different maps for steering behavior at high speeds and low speeds. The reason for this being that while high steering angles are perfectly acceptable and desirable in low speed situations such as making a right angle turn into a neighborhood street, they are not ok at high speeds, e.g. for highway driving, where sharp turns can unsettle a car's suspension.

My recovery training examples were taken at low speed where I wanted to return to the center of the track as quickly as possible. Consequently, unleasing the controller at high speed causes erratic driving behavior because it is mapping the same images to high steering angle maneuvers, whereas one should gradually correct the course to the center of the track at high speeds.

It is reasonable that the image of the upcoming road should map to speed as well. This is how good human drivers analyze the road and determine a safe speed. If you see a sharp bend coming up where the end of the road meets the side of the field of view, then you should hit the brake and slow down to safely maneuver the turn.

Left for future work are to:
* allow training of steering angle based on speed
* produce accelerator and braking output in addition to steering angle for a more realistic controller
* generalize the lane tracking to work on track 2 - which has a center lane marking in addition to different colored road edges
