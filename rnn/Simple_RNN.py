import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf = tf.compat.v1
tf.disable_v2_behavior()


def generate_data(n_samples=1000):
    X = np.random.rand(n_samples, 3)
    y = np.sum(X, axis=1)
    return X, y


# Create the synthetic dataset
X_train, y_train = generate_data()

# Define the RNN model
tf.reset_default_graph()

# Parameters
input_size = 3
hidden_size = 64
output_size = 1
learning_rate = 0.001

# Placeholders
X_placeholder = tf.placeholder(tf.float32, [None, 3])
y_placeholder = tf.placeholder(tf.float32, [None])

# RNN cell
cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)

# Initial state
initial_state = cell.zero_state(tf.shape(X_placeholder)[0], tf.float32)

# RNN layer
rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, tf.expand_dims(X_placeholder, -1), initial_state=initial_state,
                                             dtype=tf.float32)

# Output layer
output = tf.layers.dense(tf.reshape(rnn_outputs[:, -1, :], [-1, hidden_size]), output_size)

# Loss function and optimizer
loss = tf.losses.mean_squared_error(y_placeholder, tf.squeeze(output))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Training
batch_size = 32
epochs = 100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]

            _, current_loss = sess.run([optimizer, loss], feed_dict={X_placeholder: batch_X, y_placeholder: batch_y})

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {current_loss}')

    # Make predictions
    predictions = sess.run(output, feed_dict={X_placeholder: X_train})

# Print some predictions
for i in range(5):
    print(f"Actual: {y_train[i]}, Predicted: {predictions[i][0]}")
