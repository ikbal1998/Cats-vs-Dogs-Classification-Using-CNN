# Cats-vs-Dogs-Classification-Using-CNN

## 1.Introduction
The purpose of the task is to build an identity recognition system using the convolutional neural networks (CNN) technique to classify an image as either a cat or a dog. 

## 2.Project Overview

* Dataset Preparation
* Building a baseline CNN model
* Training the model  
* Improving the Model Using Data Augmentation Techniques
* Implement a Pre-trained Model (Transfer Learning)
* Testing and Inference

## 3.Results
The baseline model achieves an accuracy of 71.125%, whereas the pre-trained VGG16 model outperforms it significantly with an accuracy of 93.368%
                      
![image](https://github.com/ikbal1998/Cats-vs-Dogs-Classification-Using-CNN/assets/77022993/5e91fd04-6cfb-4e9a-a0ee-b43553673118)
![image](https://github.com/ikbal1998/Cats-vs-Dogs-Classification-Using-CNN/assets/77022993/1b2182d2-9e49-4543-9126-6f7e24abd627)

The model's accuracy is displayed on the plot with a blue line for training and a red line for testing.


Predicted Samples:
![image](https://github.com/ikbal1998/Cats-vs-Dogs-Classification-Using-CNN/assets/77022993/34153cb4-5688-449b-a6ba-71daa0162162)

## 4.Summary:
I started by developing a baseline model to gain a deeper understanding of the problem and to establish a performance benchmark. Additionally, this approach allowed for quicker model training, requiring fewer computational resources compared to training a deep pre-trained model like VGG16.

Afterwards, we introduced a pre-trained VGG16 model, which had already learned features from a large dataset like ImageNet. In the case of VGG16, we observed a significant improvement in both model performance and accuracy. 
