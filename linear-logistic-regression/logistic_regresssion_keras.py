import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf = tf.compat.v1
tf.disable_v2_behavior()

# 1. Load the data
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris = datasets.load_iris()

# Use the first 4 variables to predict the species
X, y = iris.data[:, :4], iris.target
# Split both independent and dependent variables in half for cross-validation
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.5, random_state=0)

# Train
lr = LogisticRegression()
lr.fit(train_X, train_y)

# Test
pred_y = lr.predict(test_X)


# print("Accuracy is {:.2f}".format(lr.score(test_X, test_y)))

# Keras Neural Network for Logistic Regression

# ONE-HOT encoding
def one_hot_encode_object_array(arr):
    uniques, ids = np.unique(arr, return_inverse=True)
    return to_categorical(ids, len(uniques))


# Dividing data into train and test data
train_y_ohe = one_hot_encode_object_array(train_y)
test_y_ohe = one_hot_encode_object_array(test_y)

# Creating a model
model = Sequential()
model.add(Dense(16, input_shape=(4,)))
model.add(Activation('sigmoid'))
model.add(Dense(3))
model.add(Activation('softmax'))

# Compiling the model
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

# Actual modelling
model.fit(train_X, train_y_ohe, verbose=0, batch_size=1, epochs=100)

# evaluate
score, accuracy = model.evaluate(test_X, test_y_ohe, batch_size=16, verbose=0)

# print("\nAccuracy is using keras prediction {:.2f}".format(accuracy))

print("\n Accuracy obtained by both model")
print("\n Accuracy is using keras prediction {:.2f}".format(accuracy))
print("\n Accuracy is using logistic regression {:.2f}".format(lr.score(test_X, test_y)))

# Show the trained model summary
model.summary()

# Plot the model architecture
tf.keras.utils.plot_model(model, to_file='logistic_regression_model.png', show_shapes=True)
plt.show()
