# Social Neural Networks


## Introduction
### Goal
In this work we want to check, if it is possible to *train convolutional neural networks (CNN)*
with images tagged by social networks users. Therefore this work is called *social neural networks (SNN)*.
For example, to let a network learn how cats and dogs look like we download images that are tagged with 'cat' 
and images that are tagged with 'dog' from social networks and use those images to train a CNN. To check the 
performance of the network a *well defined* dataset is used in this work. 
With well defined dataset we mean, that we can be sure that all images are correctly labeled, because lots of training samples from social 
networks contains invalid tags. In this work we used the ImageNet dataset to check the performance of the network.

### Motivation
In supervised learning a CNN is trained with labeled images. This means, that training examples
are feed forward through the network to generate a prediction using the current weights of the network. Afterwards an error between the
predictions and the labels is calculated. The error is used to adjust all learnable parameters of the network into the negative 
direction of the gradient estimate. This reduces the error on the training set and produces a more
desirable output in the next iteration. [Goodfellow et al. (2016)](#Goodfellow-et-al-2016) states, that about 5000 labeled examples per category
are needed to get an acceptable performance of the network. 

It is a timeconsuming task to create a dataset with at least 5000 labeled images per class. So in this work we want to check if it
is possible to train a CNN with images from 6 different classes (0=cat, 1=dog, 2=hamburger, 3=sushi, 4=beach, 5=rock) 
that are tagged by flickr users.

### Challenge
The challenge of this work is that lots of training examples are are invalid tagged. For example, if the network should learn 
the difference between beer and wine we download images with those tags from social networks. Sometimes images
of austrian guys are tagged with beer. Its really funny but its not helpful for our network to learn, how a beer or a wine looks like.

## System
The following image shows an overview of all different parts of the system:

![System Overview](docs/system-overview.png)

### Social Network
Flickr provides an interface to download images [[2]](#Flickr-Api-2017) from their social network. This interface is used
to download all training images.

### Training Set
The tool [datr](#Flickr-Api-2017) already provides a python library, to download images from flickr with given tags. 
This tool is used to download 10000 images per class. The training set is later split into a training and a validation set using 
3-fold cross validation.

Before images are feed into the network, random distortions per epoch are applied to all images to reduce overfitting.

### Test Set
As described above some images of the training set are invalid tagged. Therefore we use a well defined dataset
to measure the performance (f1-score) of the network. In this work we used the [Imagenet (2017)](#Imagenet-2017) dataset.
About 1300 images per class are used from this well defined dataset to measure the performance of the network.

### Network Architecture
We decided to train a network from scratch (instead of fine tuning) with input images of size 224x224x3.
The following architecture was used:

| Name          | Shape            | Output Size  |
| ------------- | ---------------- | ------------ |
| conv1_1       | 7x7x32           | 112x112x32   |
| conv1_2       | 5x5x32           | 112x112x32   |
| max_pool1     | -                | 56x56x32     |
| conv2_1       | 5x5x64           | 56x56x64     |
| conv2_2       | 5x5x80           | 56x56x80     |
| max_pool2     | -                | 28x28x80     |
| conv3_1       | 5x5x80           | 28x28x80     |
| conv3_2       | 5x5x192          | 28x28x192    |
| max_pool3     | -                | 14x14x192    |
| conv4_1       | 5x5x192          | 14x14x192    |
| conv4_2       | 5x5x192          | 14x14x192    |
| max_pool4     | -                | 7x7x192      |
| fc1 (dropout) | -                | 4704         |
| fc2 (dropout) | -                | 1176         |
| fc3 (dropout) | -                | 294          |
| out           | -                | 6            |


## Results 
After the network was trained, we measured the f1-score using images from [#Imagenet (2017)](Imagenet-2017).
The validation f1-score is about 0.60 and the test f1-score is 0.72. One hypothesis why the validation f1-score
is lower than the test f1-score is, that lots of images of the validation set are invalid tagged and therefore the network could only 
guess for those invalid images. The test set does not contain invalid labeled images and therefore the f1-score is higher.

If this hypothesis is true it should be possible to filter invalid images of the training set created from social networks with
neural networks that are trained with the same dataset. To check the hypothesis we decided to train the network a second 
time (referred to as generation 1) only on those images, that are correctly classified by generation 0 (all other images are invalid 
tagged with a probability of 70%). The idea is that generation 0 writes a list of all invalid images of the training set
into an *experience file* and generation 1 does not use those images during training. *Note: Generation 0 always filters the validation fold of the 
dataset (never the training folds). So after the network is trained on all k-folds, the whole dataset is filtered and generation 
1 can be trained.*

If the test score is almost the same for generation 1 we can 
conclude, that the first network trained with all images is able to filter invalid images from the training set although it was 
trained with those images. If the test score of generation 1 is lower, we can conclude that generation 0 removed also valid images
from the training set, the training set only becomes smaller and therefore we can conclude that the hypothesis is wrong.

After we trained generation 1 of the network, we have seen that the validation f1-score increased from 0.60 (generation 0) to 
0.72 (generation 1) and the test f1-score has not changed between both generations. This result supports the hypothesis and 
we conclude that it is possible to train a neural network with images from social networks and to use the same network to 
filter invalid images out of the training set.


## References
<a name="Goodfellow-et-al-2016">[1]</a>: Ian Goodfellow and Yoshua Bengio and Aaron Courville. *Deep Learning*, 
URL <a href="http://www.deeplearningbook.org">http://www.deeplearningbook.org</a>, 2016

<a name="Flickr-Api-2017">[2]</a>: *Flickr API*, 
URL <a hrref="https://www.flickr.com/services/api/">https://www.flickr.com/services/api/</a>, 2017

<a name="Datr-2017">[3]</a>: Download images from flickr with python, *datr*, 
URL <a hrref="http://github.com/peerdavid/datr">http://github.com/peerdavid/datr</a>, 2017

<a name="Imagenet-2017">[4]</a>: *ImageNet Dataset*, 
URL <a hrref="http://www.image-net.org/">http://www.image-net.org/</a>, 2017