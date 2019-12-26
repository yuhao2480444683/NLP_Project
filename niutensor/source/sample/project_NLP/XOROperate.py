import numpy as np

a = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])

X = np.zeros((64,6),dtype=np.float32)

for i in range(8):
    for j in range(8):
        X[i*8+j,0:3]=a[i]
        X[i*8+j,3:6]=a[j]

y1 = np.zeros((64,3),dtype=np.int)

for i in range(64):
    for j in range(3):
        if X[i][j]==X[i][j+3]:
            y1[i][j]=0
        else:
            y1[i][j]=1
            
y2 = [str(i[0])+str(i[1])+str(i[2]) for i in y1]

y3 = np.zeros(64,dtype=np.float32)

for i in range(64):
    if y2[i]=='000':
        y3[i]=0
    if y2[i]=='001':
        y3[i]=1
    if y2[i]=='010':
        y3[i]=2
    if y2[i]=='011':
        y3[i]=3
    if y2[i]=='100':
        y3[i]=4
    if y2[i]=='101':
        y3[i]=5
    if y2[i]=='110':
        y3[i]=6
    if y2[i]=='111':
        y3[i]=7

from keras.utils import to_categorical

y = to_categorical(y3)

print(X.shape)
print(y.shape)

import tensorflow as tf

n_hidden=20
learning_rate=0.01
num =50000

tf.reset_default_graph()

x_p = tf.placeholder(tf.float32, [None, 6])
y_p = tf.placeholder(tf.float32, [None, 8])

w1 = tf.get_variable(name='w1', shape=[6, n_hidden],
                             initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable(name='b1', shape=[1, n_hidden],
                             initializer=tf.zeros_initializer())
w2 = tf.get_variable(name='w2', shape=[n_hidden, 8],
                             initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable(name='b2', shape=[1, 8],
                             initializer=tf.zeros_initializer())

z1 = tf.matmul(x_p,w1) + b1
a1 = tf.nn.relu(z1)
z2 = tf.matmul(a1,w2) + b2
a2 = tf.nn.relu(z2)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=a2, labels=y_p))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(a2, 1), tf.argmax(y_p, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

predict = tf.arg_max(a2,1)

init = tf.global_variables_initializer()
sess = tf.Session() 
sess.run(init)
for i in range(num):
    sess.run(optimizer, feed_dict={x_p: X, y_p: y})
    if i%1000==0:
        train_accuracy = sess.run(accuracy, feed_dict={x_p: X, y_p: y})
        print(train_accuracy)
            #print(sess.run(tf.round(a2),feed_dict={x1: X}))
print(sess.run(predict,feed_dict={x_p:X}))












