#!/bin/bash

# 
# Create a dataset from flick. 
# Simply call ./dataset_create.sh car book face [...]
# The classes will be numbered (car=0, book=1, face=2, ...)
# and downloaded into the downloads folder. This folder can
# later be used to train the cnn
#

# Configuration
CLASSES=("$@")
DOWNLOAD_PATH="downloads"
NUM_IMAGES=1000

# Prepare downloads folder
rm -rf $DOWNLOAD_PATH
mkdir $DOWNLOAD_PATH

# First of all download all images from flickr
classId=0
for className in "${CLASSES[@]}"
do
	python dataset_download.py "$DOWNLOAD_PATH/$classId" $className $NUM_IMAGES
    let "classId=classId+1"
done

# Now preprocess images to 366x366 pixels because we crop 
# it randomly to 299x299 for the inception v3 model to 
# increase the training set size (data augmentation)
python3 dataset_preprocess.py "$DOWNLOAD_PATH/"