# coding: utf-8
import tensorflow as tf
import numpy as np
import matplotlib
import os
import matplotlib.pyplot as plt

tf.set_random_seed(777)  # reproducibility

def MinMaxScaler(data):
    ''' Min Max Normalization

    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]

    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]

    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html

    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


# train Parameters
seq_length = 15
data_dim = 1
hidden_dim = 2
output_dim = 1
learning_rate = .1
iterations = 1000

# Open, High, Low, Volume, Close
xy = np.loadtxt('data_vdc_xiaxian.csv', delimiter=',', usecols=[1])
# xy = xy[::-1]  # reverse order (chronically ordered)
xy = MinMaxScaler(xy)
xy = xy.reshape([-1, 1])

# build a dataset
dataX = []
dataY = []
for i in range(0, len(xy) - seq_length):
    _x = xy[i: i + seq_length]
    _y = xy[i + seq_length]  # Next close price
    # print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

# train/test split
train_size = int(len(dataY) * 0.7)
trainX, testX = np.array(dataX[0:train_size]), np.array(
    dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(
    dataY[train_size:len(dataY)])

#---- 构建模型 ----#
# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

# build a LSTM network
cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)

outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={
                                X: trainX, Y: trainY})
        if i % 50 == 0:
            print("[step: {}] loss: {}".format(i, step_loss))

    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    # naive_predict = np.mean(testX[:, :, [0]], axis=1) 
    naive_predict = np.mean(testX, axis=(1,2)) 
    rmse_val = sess.run(rmse, feed_dict={
                    targets: testY, predictions: test_predict})
    print("RNN RMSE: {}".format(rmse_val))
    print('Naive RMSE: {}'.format( np.sqrt(np.mean(np.square(naive_predict - testY))) ))
    # Plot predictions
    plt.plot(testY)
    plt.plot(test_predict, '--r')
    plt.plot(naive_predict, '.-g')
    plt.legend(['exact', 'rnn', 'naive'])
    plt.xlabel("Time Period")
    plt.ylabel("vdc xiaxian")
    plt.show()
