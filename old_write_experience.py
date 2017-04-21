from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np    

from inception import inception_model as inception
from sklearn import metrics

import sys
import pickle

import data_input
import params
import utils



FLAGS = params.FLAGS


def calc_metrics(sess, logits, labels_pl, images_pl, for_train_pl, dataset, should_calc_confusion_matrix=False):
    prediction = tf.argmax(logits, 1)

    # Calculate predictions of trained network
    y_true = []
    y_pred = []
    invalid_images = []
    for i in range(len(dataset.image_list)):
        feed_dict = utils.create_fine_tune_feed_data(sess, images_pl, labels_pl, for_train_pl, dataset, for_train_r=False)
        y_true.extend(feed_dict[labels_pl])
        y_pred.extend(sess.run([prediction], feed_dict=feed_dict)[0])

        if(y_true[i] != y_pred[i]):
            invalid_images.append(dataset.image_list[i])

        sys.stdout.write("  Calculating predictions ...%d%%\r" % (i * 100 / len(dataset.image_list)))
        sys.stdout.flush()
    sys.stdout.write("                                                 \r")
    sys.stdout.flush()

    # Append experience to file
    experience_file = FLAGS.generation_experience_file.format(FLAGS.generation)
    with open(experience_file, 'a') as file_handler:
        file_handler.write("\n".join(invalid_images))
        file_handler.write("\n")
    return


def append_experience():
    with tf.Session() as sess:        
        sess.run(tf.global_variables_initializer())

        # Check every image, so set batch_size to one
        FLAGS.batch_size = 1    

        # Create placeholder 
        images_pl, labels_pl, for_training_pl = utils.create_fine_tune_placeholder( 
                FLAGS.image_height, 
                FLAGS.image_width,
                FLAGS.image_depth)

        # We only write down experience on not trained data. Therefore we use the validation set
        train_file = FLAGS.generation_train_file.format(FLAGS.generation, FLAGS.cross_validation_iteration)
        with open(train_file, 'r') as file_handler:
            train_images = [line.rstrip('\n') for line in file_handler]
        
        dataset = data_input.read_image_batches_with_labels_in_blacklist_from_path(FLAGS, FLAGS.img_dir, train_images)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits, _ = inception.inference(images_pl, dataset.num_classes, 
            dropout_keep_prob=1.0, restore_logits=False, for_training=for_training_pl)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(inception.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Restore checkpoint           
        print("\n--------------------------------")                   
        checkpoint = FLAGS.generation_checkpoint.format(FLAGS.generation, FLAGS.cross_validation_iteration)
        saver.restore(sess, checkpoint)
        print('Succesfully loaded model from %s' % checkpoint)

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
            
            print("\n## Evaluate dataset")
            calc_metrics(sess, logits, labels_pl, images_pl, for_training_pl, 
                dataset, should_calc_confusion_matrix=True)

        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
        sess.close()

    print('Done.')
    return


#
# M A I N
#   
# LD_PRELOAD="/usr/lib/libtcmalloc.so" python3 fine_tune_evaluation.py
#
def main(argv=None):
    if(len(argv) > 1):
        FLAGS.generation = int(argv[1])
        FLAGS.cross_validation_iteration = int(argv[2])

    append_experience()


if __name__ == '__main__':
    tf.app.run()