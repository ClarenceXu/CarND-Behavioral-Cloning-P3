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

[image1]: ./examples/model.png 
[image2]: ./examples/center_own-data.jpg 
[image3]: ./examples/center_more-data.jpg 
[image4]: ./examples/flip-origin.jpg 
[image5]: ./examples/crop.jpg 
[image6]: ./examples/yuv.jpg 
[image7]: ./examples/loss_history.png "Flipped Image"

---

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use the Nvidia convolution neural network model. I thought this model might be appropriate because Nvidia's setup and driving condition (e.g. 3 cameras) is similar to this project. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set with ratio 0.2 

The Nvidia paper didn't mention any dropout to reduce overfitting. But I found that the model had a low mean squared error on the training set (e.g. loss: 0.0048) but a high mean squared error on the validation set (e.g. val_loss: 0.0116) 
This implied that the model might be overfitting. 

To combat the overfitting, I tried to add dropout with different combinations:
* add dropout after each convolutional layer with rate between [0.5 - 0.8]
* add dropout after the final convolutional layer with rate between [0.5 - 0.8]
* add dropout for each fully connected layer with with rate between [0.5 - 0.8]

Although the overfitting situation improves after adding dropout, the autonomous test run with the trained model performs worse.  
Thus, I decided to not add any dropout after all kinds of experiments.  

I also tried with different batch size, e.g. 32, 64, 96, 128.  And it turns out that the model performs the best with with batch size 128.     

The final step was to run the simulator to see how well the car was driving around track one.  There were 2 places with sharp curve where the vehicle might fell off the track. 
To improve the driving behavior in these cases, I recorded more data sets on sharp curves. (named more_data.zip)

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (method `get_nvidia_model()` in model.py) consisted of a convolution neural network with the following layers and layer sizes:
1. Crop the image from 160x320 to img[70:150, 0:320] (crop the upper and bottom part, which are irrelevant to the learning)
1. Convert image to 66x200
1. Convert from RGB to YUV
1. Normalization 
1. 5x5 convolutional layer
1. 5x5 convolutional layer
1. 5x5 convolutional layer
1. 3x3 convolutional layer
1. Flatten to a 1152 fully-connected layer (NOTE: in the Nvidia paper it stated 1164, which I don't understand why)
1. 100 fully connected layer
1. 50 fully connected layer
1. 10 fully connected layer
1. 1 fully connected layer

Here is a visualization of the architecture 

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

Training data consists of 3 data sets in total, prepared by calling script `get_data.sh`
* Udacity Data Set: default one
* Own Data Set: it is collected by driving on the road from both direction, and recovering from the left and right sides of the roads. 
Here is an example image of center lane driving.

![alt text][image2]

* More Data Set: it is added later on by myself at later stage to add more scenarios of the vehicle making turns on curvy roads.   
Here is an example image of driving on one of the sharp curves. 

![alt text][image3]

To augment the data set, I also flipped images and angles thinking that this would increase the data set. 
For example, here is an image that has then been flipped:

![alt text][image2]
![alt text][image4]

After the collection process, I had 11323 number of data points. 

I then preprocessed this data by crop the image with upper 70 pixel and bottom 10 pixel
See an example image after the crop 

![alt text][image5]

According to the Nvidia CNN model, I converted the color space from RGB to YUV.
See an example image after converting to YUV color space:

![alt text][image6]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model.  The validation set helped determine if the model was over or under fitting. 

I used an adam optimizer so that manually training the learning rate wasn't necessary.

I set the number of epochs to a relative high number 50, and I used callback function `early_stopping` so that the training will stop when the val_loss doesn't decrease.
See one example of the training and validation loss for each epoch 

![alt text][image7]


After several times testing with autonomous mode using various trained model file, I found that the validation loss is not a great indication of how well it drives.  

Sometimes, a trained model file with a higher loss (less epochs) performs better in the autonomous run.  

Thus I used ModelCheckpoint to save each best result while training.  And I used different saved model file to see which one drives the best.  


#### 4. Final Video
I also checked in the recorded video of the vehicle driving autonomously using the model.h5, which drove 3 laps around the track.

See the video at: ./examples/drive.mp4
