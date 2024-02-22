import numpy as np
import time

class LogisticRegression:

    def __init__(self, learning_rate=0.01, num_iterations=1000):

        self.learning_rate = learning_rate

        self.num_iterations = num_iterations

        self.weights = None

        self.bias = None



    def sigmoid(self, z):

        return 1 / (1 + np.exp(-z))


    def tanh(self, x):

        return 0.5 * (np.tanh(x) + 1)



    def initialize_parameters(self, num_features):

        self.weights = np.zeros(num_features)

        self.bias = 0



    def fit_for_sigmoid(self, X, y):

        num_samples, num_features = X.shape

        self.initialize_parameters(num_features)



        for _ in range(self.num_iterations):

            linear_model = np.dot(X, self.weights) + self.bias

            predictions = self.sigmoid(linear_model)



            # Gradient descent updates

            dw = (1 / num_samples) * np.dot(X.T, (predictions - y))

            db = (1 / num_samples) * np.sum(predictions - y)



            self.weights -= self.learning_rate * dw

            self.bias -= self.learning_rate * db



    def prediction_for_sigmoid(self, X):

        linear_model = np.dot(X, self.weights) + self.bias

        predictions = self.sigmoid(linear_model)

        return (predictions > 0.5).astype(int)



    def fit_for_tanh(self, X, y):

        num_samples, num_features = X.shape

        self.initialize_parameters(num_features)



        for _ in range(self.num_iterations):

            linear_model = np.dot(X, self.weights) + self.bias

            predictions = self.tanh(linear_model)



            # Gradient descent updates

            dw = (1 / num_samples) * np.dot(X.T, (predictions - y))

            db = (1 / num_samples) * np.sum(predictions - y)



            self.weights -= self.learning_rate * dw

            self.bias -= self.learning_rate * db



    def prediction_for_tanh(self, X):

        linear_model = np.dot(X, self.weights) + self.bias

        predictions = self.sigmoid(linear_model)

        return (predictions > 0.5).astype(int)



# Usage of Sigmoid


sigmoid_start = time.time()

X_train = np.array([[2.5, 3.5], [1.5, 2.5], [3.5, 4.5], [2.0, 2.5]])

y_train = np.array([1, 0, 1, 0])



model = LogisticRegression(learning_rate=0.01, num_iterations=1000)

model.fit_for_sigmoid(X_train, y_train)



X_test = np.array([[2.0, 3.0], [1.0, 1.5]])

predictions = model.prediction_for_sigmoid(X_test)

print("Sigmoid Predictions:", predictions)

sigmoid_end = time.time()

print("Time taken for Sigmoid prediction:", sigmoid_end - sigmoid_start)



# Usage of Tanh

tanh_start = time.time()

X_train = np.array([[2.5, 3.5], [1.5, 2.5], [3.5, 4.5], [2.0, 2.5]])

y_train = np.array([1, 0, 1, 0])



model = LogisticRegression(learning_rate=0.01, num_iterations=1000)

model.fit_for_tanh(X_train, y_train)


X_test = np.array([[2.0, 3.0], [1.0, 1.5]])

predictions = model.prediction_for_tanh(X_test)

print("Tanh Predictions:", predictions)


tanh_end = time.time()

print("Time taken for Tanh predictions:", tanh_end - tanh_start)