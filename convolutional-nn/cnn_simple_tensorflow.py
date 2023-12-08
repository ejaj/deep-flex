import tensorflow as tf
from keras import datasets
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf = tf.compat.v1
tf.disable_v2_behavior()

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


def show_images(train_images,
                class_names,
                train_labels,
                nb_samples=12, nb_row=4):
    plt.figure(figsize=(12, 12))
    for i in range(nb_samples):
        plt.subplot(nb_row, nb_row, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# show_images(train_images, class_names, train_labels)

# Data preprocessing
max_pixel_value = 255.0
train_images = train_images / max_pixel_value
test_images = test_images / max_pixel_value

# ONE-HOT encoding
train_labels = to_categorical(train_labels, len(class_names))
test_labels = to_categorical(test_labels, len(class_names))

# Model architecture implementation
INPUT_SHAPE = (32, 32, 3)
FILTER1_SIZE = 32
FILTER2_SIZE = 64
FILTER_SHAPE = (3, 3)
POOL_SHAPE = (2, 2)
FULLY_CONNECT_NUM = 128
NUM_CLASSES = len(class_names)
# Reset the default graph to avoid graph duplication
tf.reset_default_graph()

# Define placeholders for input and labels
x = tf.placeholder(tf.float32, shape=[None, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]])
y = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

# Model architecture
# Conv - 1
conv1_weights = tf.Variable(tf.truncated_normal(
    [
        FILTER_SHAPE[0],
        FILTER_SHAPE[1],
        INPUT_SHAPE[2],
        FILTER1_SIZE
    ],
    stddev=0.1)
)
conv1_biases = tf.Variable(tf.zeros([FILTER1_SIZE]))

conv1 = tf.nn.conv2d(x, conv1_weights, strides=[1, 1, 1, 1], padding='SAME') + conv1_biases
conv1 = tf.nn.relu(conv1)
pool1 = tf.nn.max_pool(conv1, ksize=[1, POOL_SHAPE[0], POOL_SHAPE[1], 1], strides=[1, 2, 2, 1], padding='SAME')
# Conv - 2
conv2_weights = tf.Variable(tf.truncated_normal(
    [
        FILTER_SHAPE[0],
        FILTER_SHAPE[1],
        FILTER1_SIZE,
        FILTER2_SIZE
    ],
    stddev=0.1)
)
conv2_biases = tf.Variable(tf.constant(0.1, shape=[FILTER2_SIZE]))
conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME') + conv2_biases
conv2 = tf.nn.relu(conv2)
pool2 = tf.nn.max_pool(conv2, ksize=[1, POOL_SHAPE[0], POOL_SHAPE[1], 1], strides=[1, 2, 2, 1], padding='SAME')

flatten = tf.reshape(pool2, [-1, pool2.get_shape().as_list()[1] * pool2.get_shape().as_list()[2] *
                             pool2.get_shape().as_list()[3]])

dense1_weights = tf.Variable(tf.truncated_normal(
    [pool2.get_shape().as_list()[1] * pool2.get_shape().as_list()[2] * pool2.get_shape().as_list()[3],
     FULLY_CONNECT_NUM], stddev=0.1))
dense1_biases = tf.Variable(tf.constant(0.1, shape=[FULLY_CONNECT_NUM]))

dense1 = tf.nn.relu(tf.matmul(flatten, dense1_weights) + dense1_biases)

output_weights = tf.Variable(tf.truncated_normal([FULLY_CONNECT_NUM, NUM_CLASSES], stddev=0.1))
output_biases = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]))

output = tf.matmul(dense1, output_weights) + output_biases

# Loss and optimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_step = optimizer.minimize(cross_entropy)

# Evaluation
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Training
BATCH_SIZE = 32
EPOCHS = 30

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(EPOCHS):
        for i in range(0, len(train_images), BATCH_SIZE):
            batch_x = train_images[i:i + BATCH_SIZE]
            batch_y = train_labels[i:i + BATCH_SIZE]

            _, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_x, y: batch_y})

        # Evaluate accuracy on the test set after each epoch
        acc = sess.run(accuracy, feed_dict={x: test_images, y: test_labels})
        print(f"Epoch {epoch + 1}/{EPOCHS}, Test Accuracy: {acc * 100:.2f}%")
