import numpy as np

X =np.array([[2, 3, 3], [4, 3, 3], [4, 3, 7], [4, 3, 6]])
y = np.array([2, 7, 1, 3])


def linear_regression(X, y, epochs=100000, learning_rate=0.0001):
    m = float(len(y))
    theta = np.random.random(X.shape[1])
    b = np.random.random()
    for i in range(epochs):
        y_predicted = np.dot(X, theta.T) + b
        loss = y_predicted - y
        for j in range(X.shape[1]):
            theta_gradient = (1 / m) * sum(X[:, j] * loss)
            theta[j] = theta[j] - (learning_rate * theta_gradient)
        b_gradient = (1 / m) * sum(loss)
        b = b - (learning_rate * b_gradient)

    return theta, b


theta, b = linear_regression(X, y)

print("theta=", theta)
print("b=", b)
print(np.dot(theta.T, X[0]) + b)
print(np.dot(theta.T, X[1]) + b)
print(np.dot(theta.T, X[2]) + b)
print(np.dot(theta.T, X[3]) + b)
print('test item', np.dot(theta.T, [1, 3, 2]) + b)





