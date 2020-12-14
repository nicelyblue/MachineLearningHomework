import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def loss_function(x, y, theta):
    m = x.shape[0]
    y_pred = predict(x, theta)
    ones = np.ones_like(y)

    first_factor = y * -np.log(y_pred)
    second_factor = (ones - y) * np.log(ones - y_pred)

    loss = np.sum(first_factor - second_factor)
    loss = (1 / m) * loss

    return loss


def predict(x, theta):
    lin_mod = x.dot(theta)
    y_predicted = 1 / (1 + np.exp(-lin_mod))

    return y_predicted


def fit(learning_rate, num_iters, x, y):
    n = x.shape[0]
    theta = np.zeros(x.shape[1])

    loss = []

    for _ in range(num_iters):
        y_predicted = predict(x, theta)
        gradient = np.dot(x.T, (y_predicted - y))
        gradient = gradient / n
        theta -= learning_rate * gradient
        loss.append(loss_function(x, y, theta))

    return theta, loss


def predict_class(x, theta):
    y_pred = predict(x, theta)
    y_predicted_cls = [1 if i > 0.5 else 0 for i in y_pred]

    return np.array(y_predicted_cls)


def multi_predict(x, theta1, theta2, theta3):
    y1 = predict(x, theta1)
    y2 = predict(x, theta2)
    y3 = predict(x, theta3)

    print(y1, end='\n')
    print(y2, end='\n')
    print(y3, end='\n')

    y_class = []

    for i in range(0, len(y1)):
        if min(y1[i], y2[i], y3[i]) == y1[i]:
            y_class.append(0)
        elif min(y1[i], y2[i], y3[i]) == y2[i]:
            y_class.append(1)
        else:
            y_class.append(2)
    return np.array(y_class)


def accuracy(y1, y2):
    w = 0
    for i in range(0, len(y1)):
        if y1[i] == y2[i]:
            w += 1

    accuracy = w / len(y1) * 100

    print('The model is ' + str(accuracy) + '% accurate')


df = pd.read_csv("multiclass_data.csv", header=None)

X = df.iloc[:, 0:5].values
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
X = np.c_[np.ones(X.shape[0]), X]

Y = df.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.95, shuffle=True)

num_iters = 1000

y_train1 = [0 if i == 0 else 1 for i in Y_train]
y_train2 = [0 if i == 1 else 1 for i in Y_train]
y_train3 = [0 if i == 2 else 1 for i in Y_train]

_, loss1 = fit(0.0001, num_iters, X_train, y_train1)
_, loss2 = fit(0.001, num_iters, X_train, y_train1)
_, loss3 = fit(0.01, num_iters, X_train, y_train1)

theta1, _ = fit(0.0001, num_iters, X_train, y_train1)
theta2, _ = fit(0.0001, num_iters, X_train, y_train2)
theta3, _ = fit(0.0001, num_iters, X_train, y_train3)

y_predicted_class = multi_predict(X_test, theta1, theta2, theta3)

print(y_predicted_class, end='\n')
print(Y_test)

accuracy(y_predicted_class, Y_test)

plot_points = range(0, num_iters)
plt.plot(plot_points, loss1, 'r', label='alpha = 0.0001')
plt.plot(plot_points, loss2, 'b', label='alpha = 0.001')
plt.plot(plot_points, loss3, 'g', label='alpha = 0.01')
plt.legend()
plt.show()
