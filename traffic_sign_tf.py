import pickle
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
import tensorflow as tf
import numpy as np

aug_data_train = './train_aug.p'
validation_file = './valid.p'
testing_file = './test.p'

with open(aug_data_train, mode='rb') as f:
    train = pickle.load(f)

X_train_aug, y_train_aug = train['features'], train['labels']


with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)

X_valid, y_valid = valid['features'], valid['labels']

with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_test, y_test = test['features'], test['labels']


### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

def preprocess_data(X, y, num_classes=43):
    """
    Normalize image data to [-1, 1], one-hot encodes classes
    """
    X = X.astype('float32')
    X = ((X - 0.) / 255.) * (1. - 0.) + 0.

    y = y.astype('int8')

    # Convert the labels from numerical labels to one-hot encoded labels
    y_onehot = np.zeros((y.shape[0], num_classes))
    for i, label in enumerate(y_onehot):
        label[y[i]] = 1.

    return X, y_onehot

X_train_aug_norm, y_train_aug_norm = preprocess_data(X_train_aug, y_train_aug)
X_valid_norm, y_valid_norm = preprocess_data(X_valid, y_valid)
X_test_norm, y_test_norm = preprocess_data(X_test, y_test)

# TF graph
tf.reset_default_graph()


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='VALID')

def initialize_variable(scope_name, var_name, shape, init=tf.zeros_initializer):
    with tf.variable_scope(scope_name) as scope:
        v = tf.get_variable(var_name, shape, initializer=init)
        scope.reuse_variables()

def get_scope_variable(scope_name, var_name):
    with tf.variable_scope(scope_name, reuse=True):
        v = tf.get_variable(var_name)
    return v

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))

## Initialize weights and biases
# Input = 32x32x3. Output = 28x28x6.
initialize_variable('conv_1', 'w', [5, 5, 3, 6], init=tf.contrib.layers.xavier_initializer(uniform=False))
initialize_variable('conv_1', 'b', [6])

## maxpool2d(x, k=2)

# Input = 14x14x6, Output = 10x10x16
initialize_variable('conv_2', 'w', [5, 5, 6, 16], init=tf.contrib.layers.xavier_initializer(uniform=False))
initialize_variable('conv_2', 'b', [16])

## maxpool2d(x, k=2)

# Input = 5x5x16, Output = 4x4x32
initialize_variable('conv_3', 'w', [2, 2, 16, 32], init=tf.contrib.layers.xavier_initializer(uniform=False))
initialize_variable('conv_3', 'b', [32])

# Input = 4x4x32. Output = 512. Flattening, use flatten(), so this is skipped
initialize_variable('ff_1', 'w', [4*4*32, 512], init=tf.contrib.layers.xavier_initializer(uniform=False))
initialize_variable('ff_1', 'b', [512])

# Input = 512, Output = 256
initialize_variable('ff_2', 'w', [512, 256], init=tf.contrib.layers.xavier_initializer(uniform=False))
initialize_variable('ff_2', 'b', [256])

# Input = 120, Output = 84
initialize_variable('ff_3', 'w', [256, 128], init=tf.contrib.layers.xavier_initializer(uniform=False))
initialize_variable('ff_3', 'b', [128])

# Input = 84, Output = 43
initialize_variable('ff_out', 'w', [128, 43], init=tf.contrib.layers.xavier_initializer(uniform=False))
initialize_variable('ff_out', 'b', [43])

keep_prob = tf.placeholder(tf.float32) # dropout: probability to keep units

def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer


    # Layer 1
    # Activation.
    x = conv2d(x, get_scope_variable('conv_1', 'w'), get_scope_variable('conv_1', 'b'))

    # Pooling
    x = maxpool2d(x, k=2)

    # Layer 2
    # Activation.
    x = conv2d(x, get_scope_variable('conv_2', 'w'), get_scope_variable('conv_2', 'b'))

    # Pooling
    x = maxpool2d(x, k=2)

    # Layer 3
    # Activation.
    x = conv2d(x, get_scope_variable('conv_3', 'w'), get_scope_variable('conv_3', 'b'))

    # Flatten.
    fc = tf.reshape(x, [-1, get_scope_variable('ff_1', 'w').get_shape().as_list()[0]])
    #fc = flatten(x)

    # Layer 3: Fully Connected.
    # Activation.
    fc = tf.nn.relu(tf.matmul(fc, get_scope_variable('ff_2', 'w')) + get_scope_variable('ff_2', 'b'))
    fc = tf.nn.dropout(fc, keep_prob)

    # Layer 4: Fully Connected.
    # Activation.
    fc = tf.nn.relu(tf.matmul(fc, get_scope_variable('ff_3', 'w')) + get_scope_variable('ff_3', 'b'))
    fc = tf.nn.dropout(fc, keep_prob)

    # Layer 5: Fully Connected.
    logits = tf.nn.relu(tf.matmul(fc, get_scope_variable('ff_out', 'w')) + get_scope_variable('ff_out', 'b'))
    return logits

# setting up training operation
LR = 0.0001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
cost_mean = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=LR)
train_operation = optimizer.minimize(cost_mean)

# setting up evaluate operation

def evaluate_logits(X_data, y_data, batch_size=512, KB=1.0, prediction=logits, truth=y):
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(truth, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    num_examples = len(X_data)
    total_accuracy = 0
    # get current session
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: KB})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# settign up logging
RUN_NAME = 'LeNetPlus'

with tf.variable_scope('log'):
    tf.summary.scalar('cost', cost_mean)
    summary = tf.summary.merge_all()

# Create log file writers to record training progress.
# We'll store training and testing log data separately.
train_writer = tf.summary.FileWriter("./pc/logs/{}".format(RUN_NAME))

### Train your model here
BATCH_SIZE = 512
EPOCHS = 80
KProb = 0.5

saver = tf.train.Saver()

# Create a summary operation to log the progress of the network
with tf.variable_scope('logging'):
    tf.summary.scalar('current_cost', cost_mean)
    summary = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train_aug_norm)

    # Create log file writers to record training progress.
    # We'll store training and testing log data separately.
    train_writer = tf.summary.FileWriter("./logs/{}".format(RUN_NAME), sess.graph)

    print("Training...")
    for epoch in range(EPOCHS):
        X_feed, y_feed = shuffle(X_train_aug_norm, y_train_aug_norm)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_feed[offset:end], y_feed[offset:end]
            _, train_summary = sess.run([train_operation, summary],
                                        feed_dict={x: batch_x, y: batch_y, keep_prob: KProb})

        train_accuracy = evaluate_logits(X_feed, y_feed)
        validation_accuracy = evaluate_logits(X_valid_norm, y_valid_norm)

        # Write the current training status to the log files (Which we can view with TensorBoard)
        train_writer.add_summary(train_summary, epoch)

        print("EPOCH {} ...".format(epoch + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print("Train Accuracy = {:.3f}".format(train_accuracy))
        print()

    saver.save(sess, './' + RUN_NAME)
    print("Model saved")