#!/bin/bash

# 
# Learn a neural network from social networks with different generations of the same model
#


# Configuration
CLASSES=("$@")
DOWNLOAD_PATH="data/train"
NUM_IMAGES=4000


function main {
    # If folder already exists, ask user if he want a new dataset
    if [ -d "$DOWNLOAD_PATH" ]; then
        read -p "A dataset already exists. Should we create a new one? (y/n) " -n 1 -r
        echo    # (optional) move to a new line
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            create_dataset
        fi
    fi

    if [ ! -d "$DOWNLOAD_PATH" ]; then
        create_dataset
    fi

    # Finished with the dataset, so train the model
    # tensorboard --logdir log &> /dev/null &

    # Train generation 0
    LD_PRELOAD="/usr/lib/libtcmalloc.so" python3 train.py 0 0
    LD_PRELOAD="/usr/lib/libtcmalloc.so" python3 train.py 0 1
    LD_PRELOAD="/usr/lib/libtcmalloc.so" python3 train.py 0 2

    # Write experience of gen 0
    python3 write_experience.py 0 0
    python3 write_experience.py 0 1
    python3 write_experience.py 0 2

    # Train generation 1
    LD_PRELOAD="/usr/lib/libtcmalloc.so" python3 train.py 1 0
    LD_PRELOAD="/usr/lib/libtcmalloc.so" python3 train.py 1 1
    LD_PRELOAD="/usr/lib/libtcmalloc.so" python3 train.py 1 2

    # Write experience of gen 1
    python3 write_experience.py 0 0
    python3 write_experience.py 0 1
    python3 write_experience.py 0 2
}


function create_dataset {
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
}


#
# Execute main
#
main