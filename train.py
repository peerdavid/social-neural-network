

#
# Social Neural Network
#
# 2017 - Peer David
#

#
# Start training with
# LD_PRELOAD="/usr/lib/libtcmalloc.so" python3 train.py
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import copy
from datetime import datetime
import os.path
import re
import time
from six.moves import xrange

import numpy as np
import tensorflow as tf

import data_input
import params
import utils

from inception import inception_model as inception
from inception.slim import slim
from shutil import copyfile

from sklearn import metrics

FLAGS = params.FLAGS


def _tower_loss(images, labels, for_training, num_classes, scope):
    """Calculate the total loss on a single tower running the ImageNet model.
    We perform 'batch splitting'. This means that we cut up a batch across
    multiple GPU's. For instance, if the batch size = 32 and num_gpus = 2,
    then each tower will operate on an batch of 16 images.
    Args:
      images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
                                        FLAGS.image_size, 3].
      labels: 1-D integer Tensor of [batch_size].
      num_classes: number of classes
      scope: unique prefix string identifying the ImageNet tower, e.g.
        'tower_0'.
    Returns:
      Tensor of shape [] containing the total loss for a batch of data
    """
    # When fine-tuning a model, we do not restore the logits but instead we
    # randomly initialize the logits. The number of classes in the output of the
    # logit is the number of classes in specified Dataset.
    restore_logits = not FLAGS.fine_tune

    # Build inference Graph.
    logits = inception.inference(images, num_classes, for_training=for_training,
                                 restore_logits=restore_logits,
                                 dropout_keep_prob=FLAGS.dropout_keep_prob,
                                 scope=scope)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    split_batch_size = images.get_shape().as_list()[0]
    inception.loss(logits, labels, FLAGS.label_smoothing, batch_size=split_batch_size)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection(slim.losses.LOSSES_COLLECTION, scope)

    # Calculate the total loss for the current tower.
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n(losses + regularization_losses, name='total_loss')

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on TensorBoard.
        loss_name = re.sub('%s_[0-9]*/' % inception.TOWER_NAME, '', l.op.name)
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(loss_name +' (raw)', l)
        tf.summary.scalar(loss_name, loss_averages.average(l))

    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss, logits


def _log_line(f, msg):
    f.write("{0} \n".format(msg))
    print(msg)


def _average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
      List of pairs of (gradient, variable) where the gradient has been averaged
      across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train():
    """Train on dataset for a number of steps."""
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)


        
        # If a generation already exists, learn from their experience about the data
        if(FLAGS.generation > 0):
            fathers_experience = FLAGS.generation - 1
            experience_file = FLAGS.generation_experience_file.format(fathers_experience)
            with open(experience_file, 'r') as file_handler:
                invalid_images = [line.rstrip('\n') for line in file_handler]
        else:
            invalid_images = []

        # Read social network images
        data_sets = data_input.read_validation_and_train_image_batches(FLAGS, FLAGS.img_dir, invalid_images)
        train_dataset = data_sets.train
        validation_dataset = data_sets.validation

        # Log images
        f = open(FLAGS.train_dir + "train_images.txt", "w")
        f.write("\n".join(train_dataset.image_list))
        f.close()

        f = open(FLAGS.train_dir + "validation_images.txt", "w")
        f.write("\n".join(validation_dataset.image_list))
        f.close()

        # Create placeholder
        images_pl, labels_pl, for_training_pl = utils.create_fine_tune_placeholder( 
                FLAGS.image_height, 
                FLAGS.image_width,
                FLAGS.image_depth)

        # Create an optimizer that performs gradient descent.
        opt = tf.train.AdamOptimizer(FLAGS.initial_learning_rate)

        input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Calculate the gradients for each model tower.
        tower_grads = []

        with tf.device('/gpu:0'):
            with tf.name_scope('gpu_0') as scope:
                # Force all Variables to reside on the CPU.
                with slim.arg_scope([slim.variables.variable], device='/cpu:0'):
                    # Calculate the loss for one tower of the ImageNet model. This
                    # function constructs the entire ImageNet model but shares the
                    # variables across all towers.
                    loss, logits = _tower_loss(images_pl,
                                            labels_pl,
                                            for_training_pl,
                                            train_dataset.num_classes,
                                            scope)

                # Reuse variables for the next tower.
                # tf.get_variable_scope().reuse_variables()

                # Retain the summaries from the final tower.
                summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                # Retain the Batch Normalization updates operations only from
                # the final tower. Ideally, we should grab the updates from
                # all towers but these stats accumulate extremely fast so we
                # can ignore the other stats from the other towers without
                # significant detriment.
                batchnorm_updates = tf.get_collection(
                    slim.ops.UPDATE_OPS_COLLECTION, scope)

                # Calculate the gradients for the batch of data on this ImageNet
                # tower.
                grads = opt.compute_gradients(loss)

                # Keep track of the gradients across all towers.
                tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = _average_gradients(tower_grads)

        # Add a summaries for the input processing and global_step.
        summaries.extend(input_summaries)

        # Calculate accuracy via sklearn => we need a prediction 
        prediction = tf.argmax(logits[0], 1)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(
                    tf.summary.histogram(var.op.name + '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # Track the moving averages of all trainable variables.
        # Note that we maintain a "double-average" of the BatchNormalization
        # global statistics. This is more complicated then need be but we employ
        # this for backward-compatibility with our previous models.
        variable_averages = tf.train.ExponentialMovingAverage(
            inception.MOVING_AVERAGE_DECAY, global_step)

        # Another possiblility is to use tf.slim.get_variables().
        variables_to_average = (tf.trainable_variables() +
                                tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variables_to_average)

        # Group all updates to into a single train op.
        batchnorm_updates_op = tf.group(*batchnorm_updates)
        train_op = tf.group(apply_gradient_op, variables_averages_op,
                            batchnorm_updates_op)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Show some training and validation images
        summaries.append(tf.summary.image('Training Images', train_dataset.images, max_outputs = 5))
        summaries.append(tf.summary.image('Validation Images', validation_dataset.images, max_outputs = 5))

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False))

        sess.run(tf.local_variables_initializer())
        sess.run(init)

        if FLAGS.fine_tune:
            assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
            variables_to_restore = tf.get_collection(
                slim.variables.VARIABLES_TO_RESTORE)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, FLAGS.pretrained_model_checkpoint_path)
            print('%s: Pre-trained model restored from %s' %
                  (datetime.now(), FLAGS.pretrained_model_checkpoint_path))

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        # Ensure, that no nodes are added to our computation graph.
        # Otherwise the learning will slow down dramatically
        tf.get_default_graph().finalize()

        log_file = open(FLAGS.train_dir + "Console.log", "w")

        try:
            for step in range(FLAGS.max_steps+1):
                if coord.should_stop():
                        break

                # Get data from queue and measure time
                start_time = time.time()
                train_feed = utils.create_fine_tune_feed_data(sess, images_pl, labels_pl, for_training_pl, 
                                                              train_dataset, for_train_r=True)
                # Now train our model
                _, loss_value = sess.run([train_op, loss], train_feed)
                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                # Print short summary every n steps
                if step % 10 == 0:
                    steps_per_epoch = train_dataset.size // train_dataset.batch_size
                    epoch = int(step / steps_per_epoch) + 1
                    num_examples_per_step = train_dataset.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    _log_line(log_file, '%s: step %d, epoch %d | loss = %.6f (%.1f examples/sec; %.3f '
                          'sec/batch)' % (datetime.now(), step, epoch, loss_value,
                                          examples_per_sec, sec_per_batch))
                
                # Write summary to tensorboard
                if step % 100 == 0:
                    log_file.flush()
                    summary_str = sess.run([summary_op], train_feed)
                    summary_writer.add_summary(summary_str[0], step)

                # Validate training and validation sets
                if step % 500 == 0:
                    validate(log_file, sess, "Validation", images_pl, labels_pl, for_training_pl, validation_dataset, 
                             summary_writer, step, prediction)
                    validate(log_file, sess, "Training", images_pl, labels_pl, for_training_pl, train_dataset, 
                             summary_writer, step, prediction)

                # Save the model checkpoint periodically.
                if step % 1000 == 0 and step != 0:
                    _log_line(log_file, "Saving checkpoint file...")
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

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


def validate(log_file, sess, dataset_name, images_pl, labels_pl, for_training_pl, dataset, summary_writer, train_step, prediction):
    """
    Measure different metrics, print a report and write the values to tensorboard
    """

    # Create test dataset
    y_true = []
    y_pred = []
    steps_per_epoch = dataset.size // dataset.batch_size
    for pred_step in xrange(steps_per_epoch):
        feed_dict = utils.create_fine_tune_feed_data(sess, images_pl, labels_pl, for_training_pl, dataset, for_train_r=False)
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

#
# M A I N
#   
def main(argv=None):

    if(len(argv) > 1):
        FLAGS.generation = int(argv[1])
        FLAGS.cross_validation_iteration = int(argv[2])
    
    # Set training dir for current fold
    FLAGS.train_dir = "{0}/generation-{1}/k-{2}/".format(
        FLAGS.train_dir,
        FLAGS.generation,
        FLAGS.cross_validation_iteration)

    # Prepare training folder
    if tf.gfile.Exists(FLAGS.train_dir):
        x = input("\nThe folder %s is not empty. Should we delete it ? (y/n) " % FLAGS.train_dir)
        #x = "y"
        if x != "y":
            print("Finished...")
            return
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)

    # Train cnn
    train()


if __name__ == '__main__':
    tf.app.run()