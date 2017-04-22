#
# Start training with
# LD_PRELOAD="/usr/lib/libtcmalloc.so" python3 train.py
#

import sys
import os
import traceback
import time
import numpy as np
from datetime import datetime

import tensorflow as tf
from sklearn import metrics

import params
import data_input
import model
import utils


FLAGS = params.FLAGS


#
# Helper functions
#          
def _create_train_loss(logits, labels):
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
        logits: Logits from inference().
        labels: Labels from distorted_inputs or inputs(). 1-D tensor
                of shape [batch_size]
    Returns:
        Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='loss_total')
    


def _create_adam_train_op(total_loss, global_step):
    """Train CIFAR-10 model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps processed.
    Returns:
        train_op: adam op for training.
    """

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(FLAGS.initial_learning_rate)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
        
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        ret = tf.no_op(name='train')

    return ret

  
def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
        total_loss: Total loss from loss().
    Returns:
        loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + " (raw)", l)
        tf.summary.scalar(l.op.name + " (avg)", loss_averages.average(l))

    return loss_averages_op


def validate(log_file, sess, dataset_name, images_pl, labels_pl, dropout_pl, dataset, summary_writer, train_step, prediction):
    """
    Measure different metrics, print a report and write the values to tensorboard
    """

    # Create test dataset
    y_true = []
    y_pred = []
    dropout_keep_propability = 1.0
    steps_per_epoch = dataset.size // dataset.batch_size
    for pred_step in range(steps_per_epoch):
        feed_dict = utils.create_feed_data(sess, images_pl, labels_pl, dropout_pl, dataset, dropout_keep_propability)
        y_true.extend(feed_dict[labels_pl])
        y_pred.extend(sess.run([prediction], feed_dict=feed_dict)[0])
        sys.stdout.write("  Calculating predictions for %s...%d%%\r" % (dataset_name, pred_step * 100 / steps_per_epoch))
        sys.stdout.flush()
    sys.stdout.write("                                                 \r")
    sys.stdout.flush()

    # Rows ~ True Labels, Cols ~ Predicted labels_pl
    # http://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    _log_line(log_file, "\n# %s\n" % dataset_name)
    _log_line(log_file, "## Confusion Matrix")
    _log_line(log_file, confusion_matrix)

    # Accuracy   
    _log_line(log_file, "\n## Summary")
    name = "Accuracy"
    val = metrics.accuracy_score(y_true, y_pred)
    log_metric(log_file, name, val, dataset_name, train_step, summary_writer)

    # Precision recall and fone score
    metric_results = metrics.precision_recall_fscore_support(y_true, y_pred)
    precision_per_class = metric_results[0]
    recall_per_class = metric_results[1]
    fone_per_class = metric_results[2]

    # Per class summary
    _log_line(log_file ,"Precision per class: %s" % ["%.2f" % v for v in precision_per_class])
    _log_line(log_file ,"Recall per class: %s" % ["%.2f" % v for v in recall_per_class])
    _log_line(log_file ,"F1-Score per class: %s" % ["%.2f" % v for v in fone_per_class])

    # Precision   
    name = "Precision"
    val = np.mean(precision_per_class)
    log_metric(log_file, name, val, dataset_name, train_step, summary_writer)

    # Precision   
    name = "Recall"
    val = np.mean(recall_per_class)
    log_metric(log_file, name, val, dataset_name, train_step, summary_writer)

    # F1 Score    
    name = "F1-Score"
    val = np.mean(fone_per_class)
    log_metric(log_file, name, val, dataset_name, train_step, summary_writer)
    _log_line(log_file, "\n")


def log_metric(log_file, name, val, dataset_name, step, summary_writer):
    _log_line(log_file, "%s %s: %0.04f" % (dataset_name, name, val))
    summary_val = tf.Summary(value=[tf.Summary.Value(tag="{0} {1}".format(name, dataset_name), simple_value=val)])
    summary_writer.add_summary(summary_val, step) 


def _log_line(f, msg):
    f.write("{0} \n".format(msg))
    print(msg)


#
# M A I N
#   
def main(argv=None):
    try:
        
        # Create log dir if not exists
        if tf.gfile.Exists(FLAGS.train_dir):
            x = input("\nThe folder %s is not empty. Should we delete it ? (y/n) " % FLAGS.train_dir)
            if x != "y":
                print("Finished...")
                return

            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)


        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():
            # ToDo: Use generation file
            invalid_images = [] 
            data_sets = data_input.read_validation_and_train_image_batches(FLAGS, FLAGS.img_dir, invalid_images)
            train_data_set = data_sets.train
            validation_data_set = data_sets.validation
            
            images_pl, labels_pl, dropout_pl = utils.create_placeholder_inputs( 
                FLAGS.image_height, 
                FLAGS.image_width,
                FLAGS.image_depth)
                    
            # Build a Graph that computes predictions from the inference model.
            # We use the same weight's etc. for the training and validation
            logits = model.inference(
                images_pl, 
                train_data_set.num_classes,
                FLAGS.image_depth,
                dropout_pl)
            
            # Add graph and placeholder to meta file
            tf.add_to_collection('logits', logits)
                            
            # Accuracy
            prediction = tf.argmax(logits, 1)

            # Add to the Graph the Ops for loss calculation.
            train_loss = _create_train_loss(logits, labels_pl)# train_data_set.labels)

            # Add to the Graph the Ops that calculate and apply gradients.
            global_step = tf.Variable(0, trainable=False)

            # Initialize optimizer
            train_op = _create_adam_train_op(train_loss, global_step)

            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=10)

            # Add tensorboard summaries
            tf.summary.image('image_train', train_data_set.images, max_outputs = 5)
            tf.summary.image('image_validation', validation_data_set.images, max_outputs = 5)

            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.summary.merge_all()
            
            # Add the variable initializer Op.
            init = tf.initialize_all_variables()
            
            # Create a session for running Ops on the Graph.
            sess = tf.Session()

            # And then after everything is built:
            # Run the Op to initialize the variables.
            sess.run(init)

            # Start the queue runners.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # Instantiate a SummaryWriter to output summaries and the Graph.
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
            
            # Ensure, that no nodes are added to our computation graph.
            # Otherwise the learning will slow down dramatically
            tf.get_default_graph().finalize()

            log_file = open(os.path.join(FLAGS.train_dir, 'console.log'), "w")

            #
            # Training loop
            #
            try:
                for step in range(FLAGS.max_steps):
                    if coord.should_stop():
                        break
                        
                    start_time = time.time()
                    
                    train_feed = utils.create_feed_data(sess, images_pl, labels_pl, dropout_pl, train_data_set, FLAGS.dropout_keep_prob)
                    _, loss_value = sess.run([train_op, train_loss], feed_dict=train_feed)
                    #_, loss_value = sess.run([train_op, train_loss])
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                    duration = time.time() - start_time

                    # Print step loss etc. to console
                    if step % 10 == 0:
                        steps_per_epoch = train_data_set.size // train_data_set.batch_size
                        epoch = int(step / steps_per_epoch) + 1
                        num_examples_per_step = train_data_set.batch_size
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = float(duration)

                        _log_line(log_file, '%s: step %d, epoch %d | loss = %.6f (%.1f examples/sec; %.3f '
                                    'sec/batch)' % (datetime.now(), step, epoch, loss_value,
                                    examples_per_sec, sec_per_batch))
                    
                    # Write summary to tensorboard
                    if step % 100 == 0:
                        log_file.flush()
                        summary_str = sess.run([summary_op], feed_dict=train_feed)
                        summary_writer.add_summary(summary_str[0], step)
                        
                    # Evaluation 
                    if step % 500 == 0:
                        validate(log_file, sess, "Validation", images_pl, labels_pl, dropout_pl, validation_data_set, 
                             summary_writer, step, prediction)
                        validate(log_file, sess, "Training", images_pl, labels_pl, dropout_pl, train_data_set, 
                             summary_writer, step, prediction)

                    # Save model checkpoint
                    if (step % 1000 == 0) or (step >= FLAGS.max_steps -1):
                        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=global_step)
            
            
            except tf.errors.OutOfRangeError:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=global_step)
                print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
            
            finally:
                # Finished
                log_file.close()
                print("\nWaiting for all threads...")
                coord.request_stop()
                coord.join(threads)
                print("Closing session...\n")
                sess.close()

    except:
        traceback.print_exc()

    finally:
        print("\nDone.\n")


if __name__ == '__main__':
    tf.app.run()