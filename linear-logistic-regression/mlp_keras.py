import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import Dense, Activation
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
lr = LinearRegression()
lr.fit(train_X, train_y)

# Test
pred_y = lr.predict(test_X)
# print("Accuracy is {:.2f}".format(lr.score(test_X, test_y)))

# Build the keras model
model = Sequential()
# 4 features in the input layer (the four flowers)
# 16 hidden units
model.add(Dense(16, input_shape=(4,)))
model.add(Activation('sigmoid'))
# 3 class in the output layer
model.add(Dense(3))
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit/Train the kears model
model.fit(train_X, train_y, verbose=1, batch_size=1, epochs=100)

# Test The modelLinearRegression

loss, accuracy = model.evaluate(test_X, test_y, verbose=0)
print("\nAccuracy is using keras prediction {:.2f}".format(accuracy))

print("\n Accuracy obtained by both model")
print("\n Accuracy is using keras prediction {:.2f}".format(accuracy))
print("\n Accuracy is using liner regression {:.2f}".format(lr.score(test_X, test_y)))

# Show the trained model summary
model.summary()

# Plot the model architecture
tf.keras.utils.plot_model(model, to_file='linear_regression_model.png', show_shapes=True)
plt.show()
