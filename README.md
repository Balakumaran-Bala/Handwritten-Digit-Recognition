# Project Overview

In this project, I intended to create an artificial neural network to recognize handwritten digits. Although there are frameworks like TensorFlow and Keras that can be implemented to solve this problem, I created the network from scratch without the implementation of any superficial frameworks. 
 
A Convolutional Neural Network (CNN) is comprised of one or more convolutional layers (often with a subsampling step) and then followed by one or more fully connected layers as in a standard multilayer neural network. 

Following the multiple layers is an iterative algorithm used to find the minimum of a function called gradient descent. Gradient descent trains our model to recognize variations within digits and thereby improves accuracy. 

# Architecture

Our Convolutional Neural Network consists of the following layers: 
- Convolution Layer: Searches for the spacial arrangement of features in the image entered   
- Pooling Layer: Down-samples the entire 28x28 matrix to a 7x7 matrix and finds the feature which has maximum frequency in each quadrant
- Feature Extraction Layer: Searches for combinations of the basic features
- Fully Connected Layer: Performs the voting system to finally zero down on the digit entered

# Convolutional Layer:

The image is initially resized to a 28 x 28 image regardless of the original image size. The convolutional layer searches for four basic features: vertical lines, horizontal lines, upward slants and downward slants. Using a 2 x 2 filter as a slide, we check for each of these features within the 28 x 28 image. This process creates four matrices for each of the basic features of size 27 x 27, where each matrix contains 0’s and 1’s. The 1’s indicate that that particular feature exists in that area of the image. 

# Pooling Layer:

Using a 4 x 4 filter with stride 3, the pooling layer finds the feature that occurs the most within each 4 x 4 matrix. This layer converts the 27 x 27 matrix into a 7 x 7 matrix, which consists of the numbers 0, 1, 2, 3 and 4. A “1” corresponds to vertical line; a “2” corresponds to a horizontal line; a “3” corresponds to an upward slant and finally, a “4” corresponds to a downward slant. The matrix is a conversion of the micro-level 27 x 27 feature matrix into a macro-level understanding of the spatial arrangement of certain features in the handwritten digit image.

# Feature Extraction Layer:

This layer searches for combinations of the basic features to obtain a better understanding of the digit. Using the 7 x 7 matrix, we search for 57 different hand-engineered features. This layer produces a feature vector of 57 elements consisting of 1’s and 0’s. A “1” denotes that a particular feature exists and likewise for a “0”.

# Fully Connected Layer:

Each of the 57 nodes are connected to 10 output nodes. The connections can be thought of as weights and are stored in a weight matrix of dimension 57 x 10. The weights are updated through gradient descent using the formula Wij := Wij - 0.01*(Softmaxi - ∆it)*FeatureListj which was derived after taking the partial derivative of our loss function, the cross-entropy function. When an image is inputted, we take the dot product of the feature list and the weight matrix to produce a vector that we call the score vector. The score vector is then put through the softmax function to produce a vector of probabilities that add up to 1. These probabilities correlate to the chance of the inputted image being each of the digits.  
