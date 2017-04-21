from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np    

from inception import inception_model as inception
from sklearn import metrics

import sys
import data_input
import params
import utils


FLAGS = params.FLAGS


def calc_metrics(sess, logits, labels_pl, images_pl, for_train_pl, dataset, should_calc_confusion_matrix=False):
    prediction = tf.argmax(logits, 1)

    # Calculate predictions of trained network
    y_true = []
    y_pred = []
    steps_per_epoch = dataset.size // dataset.batch_size
    for pred_step in range(steps_per_epoch):
        feed_dict = utils.create_fine_tune_feed_data(sess, images_pl, labels_pl, for_train_pl, dataset, for_train_r=False)
        y_true.extend(feed_dict[labels_pl])
        y_pred.extend(sess.run([prediction], feed_dict=feed_dict)[0])
        sys.stdout.write("  Calculating predictions ...%d%%\r" % (pred_step * 100 / steps_per_epoch))
        sys.stdout.flush()
    sys.stdout.write("                                                 \r")
    sys.stdout.flush()

    # Rows ~ True Labels, Cols ~ Predicted labels_pl
    # http://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    print("## Confusion Matrix")
    print(confusion_matrix)

    # Accuracy   
    print("\n## Summary")
    val = metrics.accuracy_score(y_true, y_pred)
    print("Accuracy: %0.04f" % (val))

    # Precision recall and fone score
    metric_results = metrics.precision_recall_fscore_support(y_true, y_pred)
    precision_per_class = metric_results[0]
    recall_per_class = metric_results[1]
    fone_per_class = metric_results[2]

    # Per class summary
    print("Precision per class: %s" % ["%.2f" % v for v in precision_per_class])
    print("Recall per class: %s" % ["%.2f" % v for v in recall_per_class])
    print("F1-Score per class: %s" % ["%.2f" % v for v in fone_per_class])

    # Precision   
    val = np.mean(precision_per_class)
    print("Precision: %0.04f" % (val))

    # Precision   
    val = np.mean(recall_per_class)
    print("Recall: %0.04f" % (val))

    # F1 Score    
    val = np.mean(fone_per_class)
    print("F1-Score: %0.04f" % (val))
    print("\n")

    return


#
# M A I N
#   
# LD_PRELOAD="/usr/lib/libtcmalloc.so" python3 fine_tune_evaluation.py
#
def main(argv=None):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Create placeholder 
        images_pl, labels_pl, for_training_pl = utils.create_fine_tune_placeholder( 
                FLAGS.image_height, 
                FLAGS.image_width,
                FLAGS.image_depth)

        # Load dataset to evaluate
        # ToDo: Use different path of images
        dataset = data_input.read_image_batches_with_labels_from_path(FLAGS, FLAGS.test_dir)

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
        checkpoint = FLAGS.checkpoint 
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
    return


if __name__ == '__main__':
    tf.app.run()