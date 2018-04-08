# **Traffic Sign Recognition** 

## Feng Chen, Finished on 4/8/2018

### The project is finished mostly by myself, but also partially referred to [this location](https://github.com/galenballew/Traffic-Sign-CNN)  for accuracy improvement

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

[image1]: ./submit_images/train_data_distribution_original.png "Visualization"
[image2]: ./submit_images/before_norm.png "Before Normalization"
[image3]: ./submit_images/after_norm.png "After Normalization"
[image4]: ./submit_images/after_warp.png "After Warp"
[image5]: ./submit_images/post_data_geneation.png "Post_data_geneation"
[image6]: ./submit_images/Training_Accuracy.png "Training Accuracy"
[image7]: ./submit_images/Validation_Accuracy.png "Validation Accuracy"
[image8]: ./submit_images/1.jpg "Traffic Sign 1"
[image9]: ./submit_images/2.jpg "Traffic Sign 2"
[image10]: ./submit_images/3.jpg "Traffic Sign 3"
[image11]: ./submit_images/4.jpg "Traffic Sign 4"
[image12]: ./submit_images/5.jpg "Traffic Sign 5"
[image13]: ./submit_images/6.jpg "Traffic Sign 6"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Here is the Writeup that includes all the rubric points and how I addressed each one.

You're reading it! and here is a link to my [project code](https://github.com/zypchenfeng/SDC_Term1-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the _pickle_ library to import the data, and used the _numpy_ library to calculate summary 
statistics of the traffic signs data set:

* Total 3 sets of data: Train, Validation, and Test
* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart show the 
data sample size as a function of labels. It is clearly show that the sample is 
widely biased, for all the training, validation, and test samples.

![Distribution Data][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale and at the same time normalize the data.
That is because:
1. The traffic sign usually only matters by the pattern (shape), not quite determined by color
2. The image quality is widely distributed, including contrast, brightness etc.

Here is an example of a traffic sign image before and after grayscaling.

Original data, no treatment at all.
![alt text][image2]

After Normalization and convert to Grayscale

![After Normalization and Grayscale][image3]

As a last step, I generate more data because of the non-uniform data distribution. 

To add more data to the the data set, I used the techniques that was first used in [this location](https://github.com/galenballew/Traffic-Sign-CNN)  

It inclues several techniques by using the cv2 library, including: wrap, flip, rotate etc.
Here is the grapscale images that was warpped in the data generation process.
![After Warp][image4]


The difference between the original data set and the augmented data set is the following ... 

Pre data generation
![After Warp][image1] 

Post Data Generation
![After Warp][image5]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray scale image   					| 
| Convolution 5x5x6  	| 1x1 stride, Valid padding, outputs 28x28x6 	|
| RELU					| Use Relu for activation       				|
| Max pooling	      	| k_size = 2, 2x2 stride,  outputs 14x14x6		|
| Convolution 5x5x16   	| 1x1 stride, Valid padding, outputs 10x10x16 	|
| RELU					| Use Relu for activation       				|
| Max pooling	      	| k_size = 2, 2x2 stride,  outputs 5x5x6		|
| Flatten       	    | Flatten the data before fully connected layer	|
| Fully connected		| Input nx400, output nx120						|
| RELU					| Use Relu for activation       				|
| Dropout				| Use 50% drop out to prevent overfitting		|
| Fully connected		| Input nx120, output nx84  					|
| RELU					| Use Relu for activation       				|
| Dropout				| Use 50% drop out to prevent overfitting		|
| Softmax				| Input nx84, output n*10						|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following settings:

| Description      		|     Setting	            			| 
|:---------------------:|:-------------------------------------:| 
| Learn Rate       		| 0.0001                                | 
| Epochs              	| 90                                    |
| Batch size        	| 128                              		|
| Optimizer 	      	| Adam Optimizer                        |
| Loss Operation    	| Reduce mean of cross entropy          |

I have also shuffled the data using sklearn library.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.7%
* validation set accuracy of 99.4% 
* test set accuracy of 93.2%

The training  and validation accuracy can be seen below for the 61-90 epochs
![Training Accuracy][image6]
![Validation Accuracy][image7]

I only used LeNet to build the 2 convolution 2 fully connected network. There is some
paper out there show more advanced work that can improve the test accuracy to 99%. But 
here I didn't try yet.

## The questions were not well considered during this project. Will consider later on
If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image8] ![alt text][image9] ![alt text][image10] 
![alt text][image11] ![alt text][image12]![alt text][image13]

Originally there was one 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)  | Speed limit (70km/h)   						| 
| No Passing   			| No Passing									|
| Stop					| Stop											|
| Yield	      		    | Yield					 				        |
| General caution		| General caution     							|
| Bumpy road            | Bumpy road                                    |

The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 30th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Speed limit (70km/h)  						| 
| .00     				| End of no passing								|
| .00					| No entry										|
| .00	      			| Priority road				 				    |
| .00				    | Slippery road     							|


For the second image 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Stop  						| 
| .00     				| Bumpy road								|
| .00					| Turn right ahead										|
| .00	      			| Road work			 				    |
| .00				    | Yield    							| 


For the third image 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .94         			| Yield  						| 
| .04     				| No vehicles								|
| .01					| No passing										|
| .01	      			| Ahead only				 				    |
| .00				    | Go straight or right     							| 


For the forth image 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| General caution  						| 
| .00     				| Traffic signals								|
| .00					| Pedestrians										|
| .00	      			| Road narrows on the right				 				    |
| .00				    | Go straight or left     							| 

For the fifth image 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .91         			| Bumpy road  						| 
| .09     				| Bicycles crossing							|
| .00					| Traffic signals										|
| .00	      			| No vehicles				 				    |
| .00				    | Road narrows on the right     							| 


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


