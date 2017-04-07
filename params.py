

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


#
# Data input parameter
#
flags.DEFINE_string('train_dir', "log/", 'Directory to put the log data = Output.')
flags.DEFINE_string('img_dir', 'data/train/', 'Base directory of data. Note: ptp file contains sub dir')

flags.DEFINE_integer('image_format', 0, '0 = JPEG, 1 = PNG')   
flags.DEFINE_integer('image_width', 299, 'Width in pixels of input image.')
flags.DEFINE_integer('image_height', 299, 'Height in pixels of input image.')
flags.DEFINE_integer('balance_method', 1, 'How to balance the imbalanced dataset. -1 = Nothing,  0 = Undersampling, 1 = Oversampling, 2 = Mixed Oversampling & Undersampling')
flags.DEFINE_integer('max_oversampling_degree', 100, 'Max degree of oversampling when balance method = 2')
flags.DEFINE_float('label_smoothing', 0.1, 'Smooth cross entropy by this factor if the dataset is inbalanced')

flags.DEFINE_boolean('random_distortion', True, 'Many random distortions applied to the image.')    
flags.DEFINE_integer('orig_image_width', 366, 'Width in pixels before random crop (only if random_distortion=True)')
flags.DEFINE_integer('orig_image_height', 366, 'Height in pixels before random crop (only if random_distortion=True)')

flags.DEFINE_integer('image_depth', 3, '1 = grayscale, 3 = rgb')
flags.DEFINE_integer('batch_size', 32, 'Size of a single training batch.')

flags.DEFINE_integer('num_threads', 2, 'Number of threads to fill queue of batches')

flags.DEFINE_integer('validation_size', 300, 'Number of threads to fill queue of batches')



#
# General training parameter
#
flags.DEFINE_integer('k_fold_cross_validation', 3, 'Number of iterations for k-fold cross-validation')
flags.DEFINE_integer('cross_validation_iteration', 1, 'Iteration number of cross validation')

flags.DEFINE_integer('optimizer', 1, '0 = GradientDescentOptimizer, 1 = AdamOptimizer')
flags.DEFINE_float('initial_learning_rate', 0.00001, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 5000, 'Max. number of steps to run trainer.')
flags.DEFINE_integer('num_epochs', 100000, 'Max. number of epochs to run trainer.')
flags.DEFINE_boolean('initial_accuracy', True, 'Calc accuracy at step 0?')  
flags.DEFINE_float('dropout_keep_prob', 0.8, 'Probability to keep units during training.')


#
# Finetuning parameter
#
flags.DEFINE_boolean('fine_tune', True,
                            """If set, randomly initialize the final layer """
                            """of weights in order to train the network on a """
                            """new task.""")
flags.DEFINE_string('pretrained_model_checkpoint_path', 'inception/data/inception-v3/model.ckpt-157585',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")