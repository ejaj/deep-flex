import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf = tf.compat.v1
tf.disable_v2_behavior()

d = 10
N = 100

x = tf.placeholder(tf.float32, (N, d))
y = tf.placeholder(tf.float32, (N,))
W = tf.Variable(tf.random_normal((d, 1)))
b = tf.Variable(tf.random_normal((1,)))
l = tf.reduce_sum((y - (tf.matmul(x, W) + b)) ** 2)

with tf.Session() as sess:
    tf.global_variables_initializer().run(session=sess)
    data_x = np.random.rand(N, d)
    data_y = np.random.rand(N)
    feed_dict = {x: data_x, y: data_y}
    loss_value = sess.run(l, feed_dict=feed_dict)
    print(loss_value)
