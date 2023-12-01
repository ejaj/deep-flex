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
W = tf.Variable(tf.zeros(shape=[2, 1]))
b = tf.Variable(tf.ones(shape=[1, 1]))

# 6. Declare model operations: xW+b
model = tf.add(tf.matmul(x_data, W), b)

# 7. Declare Loss function and optimizer
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model, labels=y_target))
# Declare optimizer
my_opt = tf.train.AdamOptimizer(0.02)  # learning rate = 0.02
train_step = my_opt.minimize(loss)

# 8. Initialize variables and session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# 9 Actual Prediction
prediction = tf.round(tf.sigmoid(model))
prediction_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
accuracy = tf.reduce_mean(prediction_correct)

# 10. Fit Model by using Training loops
loss_array = []
train_accuracy = []
test_accuracy = []

for i in range(1000):
    batch_size = 4
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
    batch_accuracy_train = sess.run(accuracy, feed_dict={
        x_data: predictors_vals_train,
        y_target: np.transpose([target_vals_train])
    })
    train_accuracy.append(batch_accuracy_train)
    batch_accuracy_test = sess.run(accuracy, feed_dict={
        x_data: predictors_vals_test,
        y_target: np.transpose([target_vals_test])
    })
    test_accuracy.append(batch_accuracy_test)
    if (i + 1) % 50 == 0:
        print('Loss = ' + str(bath_loss) + ' and Accuracy = ' + str(batch_accuracy_train))

# 11. check model  performance
plt.plot(loss_array, 'r-')
plt.title('Logistic Regression: Cross Entropy Loss per epoch')
plt.xlabel('Epoch')
plt.xlabel(' Cross Entropy Loss')
plt.show()
