import os
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
import numpy as np
import collections

from sklearn.cross_validation import KFold 


class DataSet(object):
    pass
    

def read_image_batches_without_labels_from_file_list(image_list, FLAGS):
    num_images = len(image_list)
    label_list = [0 for i in range(num_images)]
    data_set = _create_batches(image_list, label_list, FLAGS, num_images)
    return data_set


def read_image_batches_with_labels_from_path(FLAGS, path):
    image_list, label_list, num_classes = read_labeled_image_list(path)
    data_set = _create_batches(image_list, label_list, FLAGS)
    data_set.num_classes = num_classes
    return data_set


def read_validation_and_train_image_batches(FLAGS, path):
    print("\nReading input images from {0}".format(path))
    print("-----------------------------")
       
    # Reads pathes of images together with their labels
    image_list, label_list, num_classes = read_labeled_image_list(path)

    # ToDo: Load only images, that are not filtered by older generations

    # Split into training and validation sets for fold k
    kf = KFold(n=len(image_list), n_folds=FLAGS.k_fold_cross_validation, shuffle=True, random_state=7)
    train_index, validation_index = list(kf)[FLAGS.cross_validation_iteration]
    train_images = image_list[train_index]
    train_labels = label_list[train_index]
    validation_images = image_list[validation_index]
    validation_labels = label_list[validation_index]

    # Now shuffle images for better learning
    train_images, train_labels = _shuffle_tow_arrays_together(train_images, train_labels)
    validation_images, validation_labels = _shuffle_tow_arrays_together(validation_images, validation_labels)

    # Check if dataset is still valid
    assert all(validation_image not in train_images for validation_image in validation_images), "Some images are contained in both, validation- and training-set." 
    assert len(train_images) == len(train_labels), "Length of train image list and train label list is different"
    assert len(validation_images) == len(validation_labels), "Length of validation image list and train label list is different"

    # Create image and label batches for stochastic gradient descent
    train_data_set = _create_batches(train_images, train_labels, FLAGS)
    validation_data_set = _create_batches(validation_images, validation_labels, FLAGS)
    train_data_set.num_classes = num_classes
    validation_data_set.num_classes = num_classes

    print("Num of classes: {0}".format(num_classes))
    print("Num of training images: {0}".format(train_data_set.size))
    print("Num of training images per class: {0}".format(collections.Counter(train_labels)))
    print("Num of validation images: {0}".format(validation_data_set.size))
    print("Num of validation images per class: {0}".format(collections.Counter(validation_labels)))
    print("Batch size: {0}".format(train_data_set.batch_size))
    print("-----------------------------\n")

    assert validation_data_set.size < train_data_set.size, "More validation images than training images."

    data_sets = DataSet()
    data_sets.train = train_data_set
    data_sets.validation = validation_data_set

    return data_sets                  


def read_labeled_image_list(path):
    """Reads images and labels from file system. Create a folder for each label and put 
       all images with this label into the sub folder (you don't need a label.txt etc.)
       Note: Images can be downloaded with datr - https://github.com/peerdavid/datr
    Args:
      path: Folder, which contains labels (folders) with images.
    Returns:
      List with all filenames and list with all labels
    """
    print("Reading images from %s" % path)
    filenames = []
    labels = []
    label_dirs = [dir for dir in os.listdir(path) if os.path.isdir(os.path.join(path, dir))]
    num_classes = 0
    for label in label_dirs:
        num_classes += 1
        subdir = path + label
        for image in os.listdir(subdir):
            filenames.append("{0}/{1}".format(subdir, image))
            labels.append(int(label))
    
    assert len(filenames) == len(labels), "Supervised training => number of images and labels must be the same"
    return np.asarray(filenames), np.asarray(labels), num_classes



def _shuffle_tow_arrays_together(a, b):
    assert len(a) == len(b), "It is not possible to shuffle two lists with different len together."    

    indexes = list(range(len(a)))
    random.shuffle(indexes)
    
    shuffled_a = []
    shuffled_b = []
    for index in indexes:
        shuffled_a.append(a[index])
        shuffled_b.append(b[index])
    
    return shuffled_a, shuffled_b


def _read_images(input_queue, FLAGS):
    images_queue = input_queue[0]
    labels_queue = input_queue[1]
    files = tf.read_file(images_queue)
    
    if (FLAGS.image_format == 0):
        images = tf.image.decode_jpeg(files, channels=3)   
    else:
        images = tf.image.decode_png(files, channels=3)   

    if FLAGS.image_depth == 1:
        images = tf.image.rgb_to_grayscale(images)

    '''
        " Normalization refers to normalizing the data dimensions so that they are of approximately the same scale. 
        There are two common ways of achieving this normalization. 
        One is to divide each dimension by its standard deviation, once it has been zero-centered: (X /= np.std(X, axis = 0)). 
        Another form of this preprocessing normalizes each dimension so that the min and max along the dimension is -1 and 1 respectively. 
        It only makes sense to apply this preprocessing if you have a reason to believe that different input features have 
        different scales (or units), but they should be of approximately equal importance to the learning algorithm. 
        In case of images, the relative scales of pixels are already approximately equal (and in range from 0 to 255), 
        so it is not strictly necessary to perform this additional preprocessing step. " (http://cs231n.github.io/neural-networks-2/)"
    '''
    # We only need to scale to [0,1) if we use random_color operations
    # Scale image from [0,255] to [0,1)
    images = tf.image.convert_image_dtype(images, dtype=tf.float32)

    return images, labels_queue 


def _read_distorted_images(input_queue, FLAGS):
    images, labels = _read_images(input_queue, FLAGS)

    # Apply random distortions
    images.set_shape([FLAGS.image_height, FLAGS.image_width, FLAGS.image_depth])
    images = tf.random_crop(images, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_depth])
    images.set_shape([FLAGS.image_height, FLAGS.image_width, FLAGS.image_depth])
    images = tf.image.random_flip_left_right(images)

    # Random color operations
    images = tf.image.random_brightness(images, max_delta=32. / 255.)
    images = tf.image.random_saturation(images, lower=0.5, upper=1.5)
    images = tf.image.random_hue(images, max_delta=0.2)
    images = tf.image.random_contrast(images, lower=0.5, upper=1.5)

    # The random_* ops do not necessarily clamp.
    images = tf.clip_by_value(images, 0.0, 1.0)

    return images, labels 


def _read_validation_images(input_queue, FLAGS):
    images, labels = _read_images(input_queue, FLAGS)

    # Resize image to input size
    images = tf.image.resize_images(images, size=[FLAGS.image_height, FLAGS.image_width])
    images.set_shape([FLAGS.image_height, FLAGS.image_width, FLAGS.image_depth])

    # Flip image and crop different parts to get more validation images
    images = tf.image.random_flip_left_right(images)

    # The random_* ops do not necessarily clamp.
    images = tf.clip_by_value(images, 0.0, 1.0)

    return images, labels 


def _create_batches(image_list, label_list, FLAGS, validation=False):
    # Force all input processing onto CPU in order to reserve the GPU for
    # the forward inference and back-propagation.
    with tf.device('/cpu:0'):

        data_set = DataSet()
        data_set.size = len(image_list)
        data_set.batch_size = FLAGS.batch_size
        data_set.image_list = image_list
        data_set.label_list = label_list

        # Create input queue and shuffle after every epoch
        tf_images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
        tf_labels = ops.convert_to_tensor(label_list, dtype=dtypes.int32)
        
        input_queue = tf.train.slice_input_producer([tf_images, tf_labels], shuffle=True)

        # Read images into queue
        if validation or not FLAGS.random_distortion:
            images, labels = _read_validation_images(input_queue, FLAGS)
        else :
            images, labels = _read_distorted_images(input_queue, FLAGS)

        # Finally, rescale to [-1,1] instead of [0, 1]
        images = tf.subtract(images, 0.5)
        images = tf.multiply(images, 2.0)

        # Create batches of images with n threads
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(data_set.size * min_fraction_of_examples_in_queue)
        data_set.images, data_set.labels = tf.train.shuffle_batch(
            [images, labels], 
            num_threads=FLAGS.num_threads,
            batch_size=data_set.batch_size,
            min_after_dequeue=min_queue_examples, 
            capacity=min_queue_examples + 3 * data_set.batch_size)

    return data_set