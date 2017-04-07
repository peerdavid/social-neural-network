
import tensorflow as tf


def create_placeholder_inputs(image_height, image_width, depth):
    images_pl = tf.placeholder(tf.float32, name="images_pl", shape=(None, image_height, image_width, depth))
    labels_pl = tf.placeholder(tf.int32, name="labels_pl", shape=None)
    dropout_pl = tf.placeholder(tf.float32, name="dropout_pl", shape=None)
    return images_pl, labels_pl, dropout_pl

def create_feed_data(sess, images_pl, labels_pl, dropout_pl, data_set, dropout):
    images_r, labels_r = sess.run([data_set.images, data_set.labels])
    return {images_pl: images_r, labels_pl: labels_r, dropout_pl: dropout}

def create_placeholder_inputs_without_dropout(image_height, image_width, depth):
    images_pl = tf.placeholder(tf.float32, name="images_pl", shape=(None, image_height, image_width, depth))
    labels_pl = tf.placeholder(tf.int32, name="labels_pl", shape=None)
    return images_pl, labels_pl

def create_feed_data_without_dropout(sess, images_pl, labels_pl, data_set):
    images_r, labels_r = sess.run([data_set.images, data_set.labels])
    return {images_pl: images_r, labels_pl: labels_r}

def create_fine_tune_placeholder(image_height, image_width, depth):
    images_pl = tf.placeholder(tf.float32, name="images_pl", shape=(None, image_height, image_width, depth))
    labels_pl = tf.placeholder(tf.int32, name="labels_pl", shape=None)
    for_training_pl = tf.placeholder(tf.bool, [], name='is_training')
    return images_pl, labels_pl, for_training_pl

def create_fine_tune_feed_data(sess, images_pl, labels_pl, for_train_pl, data_set, for_train_r=False):
    images_r, labels_r = sess.run([data_set.images, data_set.labels])
    feed_dict = {images_pl: images_r, labels_pl: labels_r, for_train_pl: for_train_r}
    return feed_dict