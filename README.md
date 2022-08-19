# Deep-learning-with-Pytorch
## Introduction

This is my Deep Learning with pytorch notebook, as well as my IT 100 course project. In this notebook, I would be using the **MNIST** dataset from the Kaggle competition [Kaggle-Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer/overview "Kaggle-Digit Recognizer") to train deep learning models to recognize digit in images, and the result would be evaluated by the accuracy of the model. 

I applied the artificial neural networks (ANNs), convolutional neural networks (CNNs), and recurrent neural networks (RNNs). In conclusion, the CNNs have the best accuracy rate and works the best. The accuracy of the ANNs and RNNs both above 90%. The accuracy of RNNs appeared to be very unstable because of overfitting. 

### What is MNIST and Why

**MNIST ("Modified National Institute of Standards and Technology")** is the de facto “hello world” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As my first learning deep learning practice, I expect to have high-quality and clearly labeled datasets. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike. 

### About the Data

https://www.kaggle.com/competitions/digit-recognizer/data?select=train.csv

The data files train.csv and test.csv contain gray-scale images of hand-drawn digits, from zero through nine.

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.


### Learning resource

There are three parts of this project that I need to learn: deep learning, pytorch and the coding practice. 

- The main source of pytorch learning is [PYTORCH TUTORIALS](http://pytorch.org/tutorials/ "PYTORCH TUTORIALS"), which contains everything you need to know about the pytorch library. 

- I used the [Deep Learning for Coders with fastai and PyTorch by Jeremy Howard, Sylvain Gugger](https://www.oreilly.com/library/view/deep-learning-for/9781492045519/ "Deep Learning for Coders with fastai and PyTorch by Jeremy Howard, Sylvain Gugger") as my textbook but I manily focused on the algorithms part since I already studied a bit intro-level deep learning before. 

- I referred to many great codes from kaggle: 
 - [Pytorch Tutorial for Deep Learning Lovers](https://www.kaggle.com/code/kanncaa1/pytorch-tutorial-for-deep-learning-lovers "Pytorch Tutorial for Deep Learning Lovers")
 - [MNIST: Simple CNN , KNN (Accuracy: 100%)](https://www.kaggle.com/code/ahmed121ashraf131/mnist-simple-cnn-knn-accuracy-100-top-1 "MNIST: Simple CNN , KNN (Accuracy: 100%)")
 - [Recurrent Neural Network with Pytorch](https://www.kaggle.com/code/kanncaa1/recurrent-neural-network-with-pytorch "Recurrent Neural Network with Pytorch")
 - [PyTorch RNNs and LSTMs Explained ](https://www.kaggle.com/code/andradaolteanu/pytorch-rnns-and-lstms-explained-acc-0-99 "PyTorch RNNs and LSTMs Explained ")
