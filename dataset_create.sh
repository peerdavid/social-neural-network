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
DOWNLOAD_PATH="data/train"
NUM_IMAGES=10

# If folder already exists, ask user if he want a new dataset
if [ -d "$DOWNLOAD_PATH" ]; then
    read -p "A dataset already exists. Should we create a new one? (y/n) " -n 1 -r
    echo    # (optional) move to a new line
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
      rm -rf $DOWNLOAD_PATH  
    fi
fi

if [ ! -d "$DOWNLOAD_PATH" ]; then
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
fi