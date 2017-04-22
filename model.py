import re
import tensorflow as tf


# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'GPU0'


def conv_layer(name, input, shape, weight_decay=0.0, stride=None, visualize=None):
    if stride==None:
        stride = [1, 1, 1, 1]

    with tf.variable_scope(name) as scope:
        kernel = _variable_with_weight_decay('weights',
                                            shape=shape,
                                            stddev=5e-2,
                                            weight_decay=weight_decay)
        
        # Visualize kernel of conv1
        if visualize is not None:
            grid = put_kernels_on_grid(kernel, visualize[0], visualize[1])
            tf.summary.image("{0}/features".format(name), grid, max_outputs=1)

        conv = tf.nn.conv2d(input, kernel, stride, padding='SAME')
        biases = _get_variable('biases', [shape[3]], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv_n = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv_n)
        return conv_n


def max_pool_2x2(name, input):
    with tf.variable_scope(name) as scope: 
        return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                            padding='SAME', name=scope.name)


def fully_connected(name, input, shape, weight_decay=0.0):
    with tf.variable_scope(name) as scope:    
        weights = _variable_with_weight_decay('weights', shape=shape,
                                            stddev=0.04, weight_decay=weight_decay)
        biases = _get_variable('biases', shape[1], tf.constant_initializer(0.1))
        fc = tf.nn.relu(tf.matmul(input, weights) + biases, name=scope.name)
        _activation_summary(fc)
        return fc


def softmax(name, input, shape):
    with tf.variable_scope(name) as scope:
        weights = _variable_with_weight_decay('weights', shape,
                                            stddev=1/shape[0], weight_decay=0.0)
        biases = _get_variable('biases', [shape[1]],
                                tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(input, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)
        return softmax_linear
    

def inference(images, num_classes, depth, dropout):
#pylint: disable=maybe-no-member
    """Build the CIFAR-10 model.
    Args:
        images: Images returned from distorted_inputs() or inputs().
        num_classes: The number of classes to predict
        depth: 1 if grayscale, 3 for rgb
        dropout: Probability for droptout. Ex.: 0.75 during training and 1. for evaluation
    Returns:
        Logits.
    """

    weight_decay = 0.0001
    

    print("Building model.")

    #
    # Conv -> max_pool Layer
    #
    # Input = W1×H1×D1
    # W2=(W1−F+2P)/S+1
    # H2=(H1−F+2P)/S+1
    # D2=K

    # Input is 224x224x3
    # Note: Compromise at first layer to reduce memory usage 
    conv1_1 = conv_layer(name='conv1_1', input=images, shape=[7, 7, depth, 64], 
        weight_decay=weight_decay*2, stride=[1, 2, 2, 1], visualize=(8, 8))
    conv1_2 = conv_layer(name='conv1_2', input=conv1_1, shape=[5, 5, 64, 64], weight_decay=weight_decay)

    # Input is 112x112x64
    max_pool1 = max_pool_2x2('max_pool1', conv1_2)

    # Input is 56x56x64
    conv2_1 = conv_layer(name='conv2_1', input=max_pool1, shape=[5, 5, 64, 128], weight_decay=weight_decay)
    conv2_2 = conv_layer(name='conv2_2', input=conv2_1, shape=[5, 5, 128, 128], weight_decay=weight_decay)
    conv2_3 = conv_layer(name='conv2_3', input=conv2_2, shape=[5, 5, 128, 128], weight_decay=weight_decay)
    max_pool2 = max_pool_2x2('max_pool2', conv2_3)
   
    # Input is 28x28x128
    conv3_1 = conv_layer(name='conv3_1', input=max_pool2, shape=[5, 5, 128, 64], weight_decay=weight_decay)
    conv3_2 = conv_layer(name='conv3_2', input=conv3_1, shape=[5, 5, 64, 64], weight_decay=weight_decay)
    conv3_3 = conv_layer(name='conv3_3', input=conv3_2, shape=[5, 5, 64, 64], weight_decay=weight_decay)
    max_pool3 = max_pool_2x2('max_pool3', conv3_3)

    # Input is 14x14x64
    conv4_1 = conv_layer(name='conv4_1', input=max_pool3, shape=[5, 5, 64, 32], weight_decay=weight_decay)
    conv4_2 = conv_layer(name='conv4_2', input=conv4_1, shape=[5, 5, 32, 32], weight_decay=weight_decay)
    conv4_3 = conv_layer(name='conv4_3', input=conv4_2, shape=[5, 5, 32, 32], weight_decay=weight_decay)
    max_pool4 = max_pool_2x2('max_pool4', conv4_3)

    # Input is 7x7x32
    reshape = tf.reshape(max_pool4, [-1, 7*7*32])
    dim = reshape.get_shape()[1].value


    #
    # Fully connected layers
    #
    fc1 = fully_connected('fc1', reshape, [dim, (7*7*32) / 1], weight_decay=weight_decay)
    fc1 = tf.nn.dropout(fc1, dropout)
    fc2 = fully_connected('fc2', fc1, [(7*7*32) / 1, (7*7*32) / 2], weight_decay=weight_decay)
    fc2 = tf.nn.dropout(fc2, dropout)
    fc3 = fully_connected('fc3', fc2, [(7*7*32) / 2, (7*7*32) / 4], weight_decay=weight_decay)
    fc3 = tf.nn.dropout(fc3, dropout)
    fc4 = fully_connected('fc4', fc3, [(7*7*32) / 4, (7*7*32) / 16], weight_decay=weight_decay)
    fc4 = tf.nn.dropout(fc4, dropout)
    fc5 = fully_connected('fc5', fc4, [(7*7*32) / 16, (7*7*32) / 32], weight_decay=weight_decay)
    fc5 = tf.nn.dropout(fc5, dropout)

    #
    # Force output to represent the propability
    #
    softmax_linear = softmax('softmax_linear', fc5, [49, num_classes])

    return softmax_linear
    

def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
        x: Tensor
    Returns:
        nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _get_variable(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
    Returns:
        Variable Tensor
    """
    return tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)


def _variable_with_weight_decay(name, shape, stddev, weight_decay):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        weight_decay: add L2Loss weight decay multiplied by this float. If None, weight
                      decay is not added for this Variable.
    Returns:
        Variable Tensor
    """
    dtype = tf.float32 #tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _get_variable(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if weight_decay is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var
  
 
# https://gist.github.com/kukuruza/03731dc494603ceab0c5
# https://github.com/tensorflow/tensorflow/issues/908
def put_kernels_on_grid (kernel, grid_X, grid_Y, pad=1):
    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.
    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)
    
    Return:
      Tensor of shape [(Y+pad)*grid_Y, (X+pad)*grid_X, NumChannels, 1].
    '''
    # pad X and Y
    x1 = tf.pad(kernel, tf.constant( [[pad,0],[pad,0],[0,0],[0,0]] ))

    # X and Y dimensions, w.r.t. padding
    Y = kernel.get_shape()[0] + pad
    X = kernel.get_shape()[1] + pad
    Z = kernel.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, Z]))
    
    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, Z]))
    
    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.summary.image order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 1]
    x_min = tf.reduce_min(x7)
    x_max = tf.reduce_max(x7)
    x8 = (x7 - x_min) / (x_max - x_min)

    return x8
