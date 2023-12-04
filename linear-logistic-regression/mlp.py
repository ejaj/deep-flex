import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf = tf.compat.v1
tf.disable_v2_behavior()
# 1. Load the data
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris = datasets.load_iris()
# X is Sepal.Length and Y is Petal Length
predictors_vals = np.array(
    [predictors[0:2] for predictors in iris.data]
)
# print(predictors_vals)

target_vals = np.array(
    [1. if predictor == 0 else 0. for predictor in iris.target]
)

# 2. Split Data into train and test 80%-20%
predictors_vals_train, predictors_vals_test, target_vals_train, target_vals_test = train_test_split(predictors_vals,
                                                                                                    target_vals,
                                                                                                    test_size=0.2,
                                                                                                    random_state=12)

# 3. Normal if needed
# 4. Initialize placeholders that will contain predictors and target

x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 5. Create variables (Weight and Bias) that will be tuned up
hidden_layer_nodes = 10
# For first layer
A1 = tf.Variable(tf.ones(shape=[2, hidden_layer_nodes]))
b1 = tf.Variable(tf.ones(shape=[hidden_layer_nodes]))
# For second layer
A2 = tf.Variable(tf.ones(shape=[hidden_layer_nodes, 1]))
b2 = tf.Variable(tf.ones(shape=[1]))

# 6. Define model structure
hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data, A1), b1))
final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output, A2), b2))

# 7. Declare loss function (MSE) and optimizer
loss = tf.reduce_mean(tf.square(y_target - final_output))
my_opt = tf.train.AdamOptimizer(0.02)
train_step = my_opt.minimize(loss)

# 8. Initialize variables and session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# 9. Training loop
loss_array = []
test_loss = []
batch_size = 20
for i in range(500):

    batch_index = np.random.choice(len(predictors_vals_train), size=batch_size)
    batch_x = predictors_vals_train[batch_index]
    batch_y = np.transpose([target_vals_train[batch_index]])

    sess.run(train_step, feed_dict={
        x_data: batch_x,
        y_target: batch_y
    })
    # loss function per epoc
    bath_loss = sess.run(loss, feed_dict={
        x_data: batch_x,
        y_target: batch_y
    })
    loss_array.append(bath_loss)

    # accuracy for each epoch for train

    test_temp_loss = sess.run(loss, feed_dict={
        x_data: predictors_vals_test,
        y_target: np.transpose([target_vals_test])
    })
    test_loss.append(np.sqrt(test_temp_loss))
    if (i + 1) % 50 == 0:
        print('Loss = ' + str(bath_loss))

# 10. Check model performance
# Plot loss (mean squre error) over time

plt.plot(loss_array, 'o-', label="Train loss")
plt.plot(test_loss, 'r--', label="Test Loss")
plt.title("Loss per Generation")
plt.legend(loc="lower left")
plt.xlabel("Generation")
plt.ylabel("Loss")
plt.show()
