#coding:utf-8
import tensorflow as tf
import numpy as np
def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        return outputs

def compute_accuraty(v_xs,v_ys):
    global prediction
    y_pre=sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return  result
#手写体数字库（MNIST库)
#数据中包含55000张训练图片，每张图片的分辨率是28×28，所以我们的训练网络输入应该是28×28=784个像素数据。
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

xs=tf.placeholder(tf.float32,[None,784])
ys=tf.placeholder(tf.float32,[None,10])

prediction=add_layer(xs,784,10,activation_function=tf.nn.softmax)

#loss funciton
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),axis=1))

#train
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess=tf.Session()
sess.run(tf.global_variables_initializer())



for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)  # 现在开始train，每次只取100张图片
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i %50 ==0:
        print(compute_accuraty(mnist.test.images,mnist.test.labels))
