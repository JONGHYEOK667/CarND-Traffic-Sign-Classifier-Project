# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_fig/1.Composition_of_data.jpg "Composition of data"
[image2]: ./output_fig/2.Sample_train_Image.jpg "Sample train Image"
[image3]: ./output_fig/3.Train_history.jpg "Train histroy"
[image4]: ./output_fig/placeholder.png "Traffic Sign 1"
[image5]: ./output_fig/placeholder.png "Traffic Sign 2"
[image6]: ./output_fig/placeholder.png "Traffic Sign 3"
[image7]: ./output_fig/placeholder.png "Traffic Sign 4"
[image8]: ./output_fig/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [Jonghyeok's project code](https://github.com/JONGHYEOK667/Udacity_SelfDrivingCar_P3/blob/main/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a hist plot of Train / Valid / Test dataset composition   
As you can see in the graph below, the data distribution for the labels is uneven. However, it can be seen that the density of Train / Valid / Test data for each label is similar.  
Therefore, it seems to be possible to check the difference in the learning result according to the relative ratio of each label (the amount of data for each label).

![alt text][image1]  

And, here is a sample Traing data  
Because of the low image size (32x32), it is bad resolution
Also, there are many symbols, numbers in the red edged triangle of dataset.  
Therefore, it is necessary to check the learning results in the data set with low resolution.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


I normalized the image data :   
`X_train = X_train.astype(np.float32)`  
`X_train_norm = (X_train)/255`  

`X_valid = X_valid.astype(np.float32)`  
`X_valid_norm = (X_valid)/255`  

`X_test = X_test.astype(np.float32)`  
`X_test_norm = (X_test)/255`  



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32, RELU activation|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
| Drop out      	| rate = 0.35,  outputs 16x16x32 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x64, RELU activation|
| Max pooling	      	| 2x2 stride,  outputs 8x8x64 				|
| Drop out      	| rate = 0.35,  outputs 8x8x64 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x128, RELU activation|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 8x8x256, RELU activation|
| Max pooling	      	| 2x2 stride,  outputs 3x3x256 				|
| Drop out      	| rate = 0.35,  outputs 3x3x256 				|
| Flatten     	| outputs 2304 				|
| Fully connected		| outputs 512      									|
| Drop out      	| rate = 0.35,  outputs 512 				|
| Fully connected		| outputs 256      									|
| Drop out      	| rate = 0.35,  outputs 256 				|
| Fully connected		| outputs 128      									|
| Drop out      	| rate = 0.35,  outputs 128 				|
| Fully connected		| outputs 43      									|
| Softmax				|outputs 43      									|
|:---------------------:|:---------------------------------------------:| 
|	Total parameters					|					1,738,347							|
|			Trainable parameters					|						1,738,347									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

* Optimizer : Adam (w/ default setting)  
* Batch size : 1024 (Choose a Batch size that minimizes the time takes to run an epoch)  
* Number of epochs : 80 (but, Early stopping callback is applied)
* Early stopping : patience=5, monitor='val_accuracy'


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.02%
* validation set accuracy of 94.88%
* test set accuracy of 94.35%

![alt text][image3]

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen :   
Use the LeNet architecture discussed previously
* What were some problems with the initial architecture :  
Lower accuracy compared to the project goal
* How was the architecture adjusted and why was it adjusted : 
1. Stacking convolution layer, fully connected layer (Commonly available methed to enhance the preformance)  
2. Add Dropout layer. (Prevent model overfitting)
* Which parameters were tuned :  
Dropout rate (Adjusted for model overfitting and accuracy) 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


