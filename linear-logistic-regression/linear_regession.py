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
    [predictors[0] for predictors in iris.data]
)
# print(predictors_vals)

target_vals = np.array(
    [predictors[2] for predictors in iris.data]
)

# 2. Split Data into train and test 80%-20%
x_train, x_test, y_train, y_test = train_test_split(predictors_vals, target_vals, test_size=0.2, random_state=12)

# 3. Normal if needed
# 4. Initialize placeholders that will contain predictors and target

predictor = tf.placeholder(shape=[None, 1], dtype=tf.float32)
target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 5. Create variables (Weight and Bias) that will be tuned up
A = tf.Variable(tf.zeros(shape=[1, 1]))
b = tf.Variable(tf.ones(shape=[1, 1]))

# 6. Declare model operations: Ax+b
model_output = tf.add(tf.matmul(predictor, A), b)

# 7. Declare Loss function and optimizer
loss = tf.reduce_mean(tf.abs(target - model_output))
# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

# 8. Initialize variables and session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# 9. Fit Model by using Training loops
loss_array = []
batch_size = 40

for i in range(200):
    rand_rows = np.random.randint(0, len(x_train) - 1, size=batch_size)
    batch_x = np.transpose([x_train[rand_rows]])
    batch_y = np.transpose([y_train[rand_rows]])
    sess.run(train_step, feed_dict={
        predictor: batch_x,
        target: batch_y
    })
    bath_loss = sess.run(loss, feed_dict={
        predictor: batch_x,
        target: batch_y
    })
    loss_array.append(bath_loss)
    if (i + 1) % 50 == 0:
        print('Step Number ' + str(i + 1) + ' A = ' + str(sess.run(A)) + ', b = ' + str(sess.run(b)))
        print('L1 Loss = ' + str(bath_loss))

[slop] = sess.run(A)
[y_intercept] = sess.run(b)

# 10. Check and Display the result on test data
loss_array = []
batch_size = 30

for i in range(100):
    rand_rows = np.random.randint(0, len(x_test) - 1, size=batch_size)
    batch_x = np.transpose([x_test[rand_rows]])
    batch_y = np.transpose([x_test[rand_rows]])
    sess.run(train_step, feed_dict={
        predictor: batch_x,
        target: batch_y
    })
    bath_loss = sess.run(loss, feed_dict={
        predictor: batch_x,
        target: batch_y
    })
    loss_array.append(bath_loss)
    if (i + 1) % 50 == 0:
        print('Step Number ' + str(i + 1) + ' A = ' + str(sess.run(A)) + ', b = ' + str(sess.run(b)))
        print('L1 Loss = ' + str(bath_loss))

[slop] = sess.run(A)
[y_intercept] = sess.run(b)

# Original Data and plot
plt.plot(x_test, y_test, 'o', label="Actual Data")
test_fit = []
for i in x_test:
    test_fit.append(slop * i + y_intercept)

# predicted values and plot
plt.plot(x_test, test_fit, 'r-', label="Predicted line", linewidth=3)
plt.legend(loc='lower right')
plt.title('Petal Length vs Sepal Length')
plt.ylabel('Petal Length')
plt.xlabel('Sepal Length')
plt.show()

# plot loss over time
plt.plot(loss_array, 'r-')
plt.title('L1 Loss per loop')
plt.xlabel('Loop')
plt.ylabel('L1 Loss')
plt.show()
