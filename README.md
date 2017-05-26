# Social Neural Networks


## Introduction
### Goal
In this work we want to check, if it is possible to *train convolutional neural networks (CNN)*
with images that are tagged by users in social networks. Therefore this work is called *social neural networks (SNN)*.
For example, we download images that are tagged with 'cat' and images that are tagged with 'dog' and use those images, to train a CNN.
Afterwards we want to check the performance of the network using a *well defined* dataset. With well defined dataset we mean, 
that all images are correctly labeled (because lots of images from social networks contains invalid tags). 
In this work we used the ImageNet dataset to check the performance of the network.

### Motivation
In supervised learning a CNN is trained with labeled images. This means, that training examples
are feed forward through the network to generate a prediction using the current weights of the network. Afterwards an error between the
predictions and the labels is calculated. The error is used to adjust all learnable parameters of the network into the negative 
direction of the gradient estimate. This reduces the error on the training set and produces a more
desirable output in the next iteration. [Goodfellow et al. (2016)](#Goodfellow-et-al-2016) states that about 5.000 labeled examples per category
are needed to get an acceptable performance of the network. 

The creation of datasets with at least 5.000 labeled images per class is a time consuming task. So in this work we used
images tagged by flickr users to create a dataset for 6 different classes. 

### Challenge
The challenge of this work is, that lots of examples are are invalid tagged. For example an image of 
an austrian guy is tagged with beer. Its funny but its not helpful for our network to learn, how a beer
looks like.


## Data
ToDo: Describe and cite datr


## Network Architecture
ToDo: Describe why we not used a pre-trained inception model


### Multiple generations
ToDo: Describe the idea of multiple generations

## Metrics and Test Set
ToDo: Describe why we used imagenet 


## Results 


## Future works


## References
<a name="Goodfellow-et-al-2016">[1]</a>: Ian Goodfellow and Yoshua Bengio and Aaron Courville. *Deep Learning*, 
URL <a href="http://www.deeplearningbook.org">http://www.deeplearningbook.org</a>, 2016