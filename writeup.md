# **Traffic Sign Recognition** 

## Writeup

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

[image1]: ./writeup_image/Sample_distribution_within_different_data_sets.png "Visualization"
[image2]: ./writeup_image/gray.png "Grayscaling"
[image3]: ./writeup_image/augmented_image.png "augmented_image"
[image4]: ./writeup_image/learning_curve.png "learning_curve"
[image5]: ./writeup_image/test_image.png "test_image"
[image6]: ./writeup_image/test_image_result.png "test_image_result"
[image7]: ./writeup_image/top5_prob.png "top5_prob"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

#### You're reading it! and here is a link to my [project code](https://view5f1639b6.udacity-student-workspaces.com/notebooks/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier_a%20.ipynb)


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribute 

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because model need image with deepth=1.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because the image data should be normalized so that the data has mean zero and equal variance

analysis the the above statistical distribution figure, I decided to generate additional data because , because the distribution of each class is very uneven. So perform data augmentation so that each calss has the same distribution

To add more data to the the data set, I used the ImageDataGenerator to generator more image from the original images with tricks like rotation, zoom, shift.

Here is ten examples of random choiced augmented image:

![alt text][image3]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x20 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x60 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x60 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 12x12x160	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x150 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x150 				    |
| Fully connected		| input 3750, outputs 600                    	|
| RELU					|												|
| Drop out				|keep_prob = 0.5								|
| Fully connected		| input 600, outputs 200                    	|
| RELU					|												|
| Drop out				|keep_prob = 0.5								|
| LOGITS				| output 43    									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer optimizer,batch size=128, epochs = 20, learning rate = 0.0005.
the learning rate fist choice 0.001, but the result fluctuates heavily, so I reduce the value to 0.0005.The model almost come to the maximum value, around epoch=20, so I set the epochs=20 .

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* train set accuracy of 0.999 
* validation set accuracy of 0.971 
* test set accuracy of 0.952

If an iterative approach was chosen:
* The first architecture that was tried is the LENET which I study in our lesson.
* The LENET has high efficiency but the validation set accuracy is low than 0.9, which means that this model is underfitting.
* It may be that too few layers lead to underfitting，so I added two convolution lays. in order to reduce parameters I also added two maxpool after relu.
In order to avoid overfitting, and to improve the generalization ability of the model, I also added dropout layer after each full connected layer.
* The learning rate were tuned,because the result fluctuates heavily, so I reduce the value from 0.001 to 0.0005.
* convolution work well at Image Identification field, our project is in this field, so we can use convolution layer to solve this problem.dropout layer can effectively avoid overfitting.

Here are the final model learning curve,from the curve we can see that both the train and valid accuracy converging to a high value, and the gap between them is small, so I think the model is fit very well, no overfitting or underfitting. From the last two epochs, there has been a trend of overfitting,so we should not set the number of epoch too large, about 20 is appropriate.

![alt text][image4]

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web:

![alt text][image5]

In the above ten images, there are some characters that may bring challenges to the net.such as the number one and four image are too dark and the contrast is not obvious, the number six and ten image have some degree of jitteriness, even the human eye is difficult to distinguish.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image6]

The model was able to correctly guess 10 of the 10 traffic signs, which gives an accuracy of 100%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the end cell of the Ipython notebook.

![alt text][image7]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


