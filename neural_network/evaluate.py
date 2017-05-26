import sys
import traceback
import numpy as np
import tensorflow as tf
from sklearn import metrics

import params
import utils
import data_input


FLAGS = params.FLAGS


def calc_metrics(sess, prediction, labels_pl, images_pl, dropout_pl, dataset, should_calc_confusion_matrix=False):
    # Calculate predictions of trained network
    y_true = []
    y_pred = []
    steps_per_epoch = dataset.size // dataset.batch_size
    for pred_step in range(steps_per_epoch):
        feed_dict = utils.create_feed_data(sess, images_pl, labels_pl, dropout_pl, dataset, 1.0)
        y_true.extend(feed_dict[labels_pl])
        y_pred.extend(sess.run([prediction], feed_dict=feed_dict)[0])
        sys.stdout.write("  Calculating predictions ... %d%%\r" % (pred_step * 100 / steps_per_epoch))
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
def main(argv=None):
    try:
        # Create a session for running Ops on the Graph.
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            # Load all images from disk
            dataset = data_input.read_image_batches_with_labels_from_path(FLAGS, FLAGS.test_dir)

            # Inference model
            print("Restore graph from meta file {0}.meta".format(FLAGS.checkpoint))
            saver = tf.train.import_meta_graph("{0}.meta".format(FLAGS.checkpoint))

            print("\nRestore session from checkpoint {0}".format(FLAGS.checkpoint))
            saver.restore(sess, FLAGS.checkpoint)

            graph = tf.get_default_graph()
            
            # Load model and placeholder
            logits = tf.get_collection('logits')[0] 
            images_pl = graph.get_tensor_by_name('images_pl:0')
            labels_pl = graph.get_tensor_by_name('labels_pl:0')
            dropout_pl = graph.get_tensor_by_name('dropout_pl:0')

            # Prediction used for confusion matrix
            prediction = tf.argmax(logits, 1)
            
            try:
                # Start the queue runners.
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                # Run evaluation
                print("\n# Evaluate dataset")
                calc_metrics(sess, prediction, labels_pl, images_pl, dropout_pl, 
                    dataset, should_calc_confusion_matrix=True)

            finally:
                input("Press any key to finish...")
                print("\nWaiting for all threads...")
                coord.request_stop()
                coord.join(threads)

    except:
        traceback.print_exc()

    finally:
        print("\nDone.\n")



if __name__ == '__main__':
    tf.app.run()