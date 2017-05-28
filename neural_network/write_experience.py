import sys
import traceback
import numpy as np
import tensorflow as tf
from sklearn import metrics

import params
import utils
import data_input


FLAGS = params.FLAGS


def _log_all_wrong_predictions(sess, prediction, labels_pl, images_pl, dropout_pl, dataset):
    # Calculate predictions of trained network
    y_true = []
    y_pred = []
    invalid_images = []
    for i in range(len(dataset.image_list)):
        feed_dict = utils.create_feed_data(sess, images_pl, labels_pl, dropout_pl, dataset, 1.0)
        y_true.extend(feed_dict[labels_pl])
        y_pred.extend(sess.run([prediction], feed_dict=feed_dict)[0])

        if(y_true[i] != y_pred[i]):
            invalid_images.append(dataset.image_list[i])

        sys.stdout.write("  Calculating predictions ... %d%%\r" % (i * 100 / len(dataset.image_list)))
        sys.stdout.flush()
    sys.stdout.write("                                                  \r")
    sys.stdout.flush()

    # Append experience to file
    experience_file = FLAGS.generation_experience_file.format(FLAGS.generation)
    with open(experience_file, 'a') as file_handler:
        file_handler.write("\n".join(invalid_images))
        file_handler.write("\n")

    return



def _append_experience():
    try:
        # Create a session for running Ops on the Graph.
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Check every image, so set batch_size to one
            FLAGS.batch_size = 1    

            # We only write down experience on not trained data. Therefore we use the validation set
            train_file = FLAGS.generation_train_file.format(FLAGS.generation, FLAGS.cross_validation_iteration)
            with open(train_file, 'r') as file_handler:
                train_images = [line.rstrip('\n') for line in file_handler]
            
            dataset = data_input.read_image_batches_with_labels_in_blacklist_from_path(FLAGS, FLAGS.train_dir, train_images)

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

                # Write experience
                print("\n# Writing experience file for gen. {0} iteration {1} ".format(
                    FLAGS.generation,
                    FLAGS.cross_validation_iteration
                ))

                _log_all_wrong_predictions(sess, prediction, labels_pl, images_pl, dropout_pl, dataset)

            finally:
                input("Press any key to finish...")
                print("\nWaiting for all threads...")
                coord.request_stop()
                coord.join(threads)

    except:
        traceback.print_exc()

    finally:
        print("\nDone.\n")


#
# M A I N
#
def main(argv=None):
    if(len(argv) > 1):
        FLAGS.generation = int(argv[1])
        FLAGS.cross_validation_iteration = int(argv[2])
        FLAGS.checkpoint = FLAGS.generation_checkpoint.format(FLAGS.generation, FLAGS.cross_validation_iteration)

    _append_experience()



if __name__ == '__main__':
    tf.app.run()