import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.neural_network import MLPRegressor
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


def neural_net_model(X_data,input_dim):
    hidden_layer_nodes = 32
    A1 = tf.Variable(tf.random_normal(mean=0.5, stddev=0.5, shape=[input_dim, hidden_layer_nodes]))  # inputs -> hidden nodes
    b1 = tf.Variable(tf.random_normal(mean=0.5, stddev=0.5, shape=[hidden_layer_nodes]))  # one biases for each hidden node
    A2 = tf.Variable(tf.random_normal(mean=0.5, stddev=0.5, shape=[hidden_layer_nodes, 1]))  # hidden inputs -> 1 output
    b2 = tf.Variable(tf.random_normal(mean=0.5, stddev=0.5, shape=[1]))  # 1 bias for the output

    # Declare model operations
    hidden_output = tf.nn.relu(tf.add(tf.matmul(X_data, A1), b1))
    final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output, A2), b2))

    return final_output

xs = tf.placeholder("float")
ys = tf.placeholder("float")

output = neural_net_model(xs, 3)

cost = tf.reduce_mean(tf.square(output - ys))
train = tf.train.AdamOptimizer(0.001).minimize(cost)
# train = tf.train.GradientDescentOptimizer(0.005).minimize(cost)

train_X = [[[0.1,0.2,0.3]], [[0.4,0.1,0.8]], [[0.2,0.1,0.7]], [[0.1,0.2,0.8]]]
# train_X = [[[0.1,0.2,0.6]], [[0.4,0.1,0.9]], [[0.2,0.1,0.8]], [[0.1,0.2,20.0]]]
train_y = [10, 20, 18, 19]
epoch_costs = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    for i in range(1000):
        # sess.run([cost,train],feed_dict={xs:X_train, ys:y_train})
        for j in range(len(train_X)):
            sess.run(train, feed_dict={
                xs: train_X[j], ys:train_y[j]})
            epoch_cost = sess.run(cost, feed_dict={
                xs: train_X[j], ys: train_y[j]})
            epoch_costs.append(epoch_cost)
        print('Epoch :', i, 'Cost :', np.average(epoch_costs))