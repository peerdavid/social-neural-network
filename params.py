

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


#
# Preprocessing
#
flags.DEFINE_string('img_to_preprocess', 'data/test/', 'Images to preprocess. Folder structure must be the same as log_dir')
#flags.DEFINE_string('img_to_preprocess', 'data/train/', 'Images to preprocess. Folder structure must be the same as log_dir')

#
# Data input
#
flags.DEFINE_string('log_dir', "log", 'Directory to put the log data = Output.')
flags.DEFINE_string('test_dir', 'data/test/', 'Base directory of data = Input (The folder name is the class id)')
flags.DEFINE_string('train_dir', 'data/train/', 'Base directory of data = Input (The folder name is the class id)')

flags.DEFINE_integer('image_format', 0, '0 = JPEG, 1 = PNG')   
flags.DEFINE_integer('orig_image_width', 366, 'Width in pixels before random crop (only if random_distortion=True)')
flags.DEFINE_integer('orig_image_height', 366, 'Height in pixels before random crop (only if random_distortion=True)')
flags.DEFINE_integer('image_width', 224, 'Width in pixels of input image.')
flags.DEFINE_integer('image_height', 224, 'Height in pixels of input image.')

flags.DEFINE_integer('batch_size', 32, 'Size of a single training batch.')
flags.DEFINE_boolean('random_distortion', True, 'Many random distortions applied to the image.')    
flags.DEFINE_integer('image_depth', 3, '1 = grayscale, 3 = rgb')
flags.DEFINE_integer('num_threads', 2, 'Number of threads to fill queue of batches')

flags.DEFINE_integer('k_fold_cross_validation', 3, 'We wan a k-Fold cross validation.')
flags.DEFINE_integer('cross_validation_iteration', 1, 'Current cross validation iteration. 0 ... n-1')


#
# Training
#
flags.DEFINE_float('initial_learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 12000, 'Max. number of steps to run trainer.')
flags.DEFINE_integer('num_epochs', 100000, 'Max. number of epochs to run trainer.')
flags.DEFINE_float('dropout_keep_prob', 0.8, 'Probability to keep units during training.')


#
# Generation of network
#
flags.DEFINE_integer('generation', 1, 'Current generation')
flags.DEFINE_string('generation_checkpoint', "log/generation-{0}/k-{1}/model.ckpt-2000", 'Use this checkpoint file to restore the values')
flags.DEFINE_string('generation_experience_file', "log/generation-{0}/experience.txt", "WHat the gen x learned about the classes")
flags.DEFINE_string('generation_train_file', "log/generation-{0}/k-{1}/train_images.txt", "WHat the gen x learned about the classes")


#
# Evaluation
#
flags.DEFINE_string('checkpoint', "log/generation-1/k-0/model.ckpt-12000", 'Use this checkpoint file to restore the values')