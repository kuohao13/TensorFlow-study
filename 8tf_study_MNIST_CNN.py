#coding:utf-8
#练习TensorFlow官方的列子，使用CNN来识别MNIST手写数据
#卷积使用1步长（stride size），0边距（padding size）的模板，保证输出和输入是同一个大小。
# 我们的池化用简单传统的2x2大小的模板做max pooling
#两个卷积层，1个全连接层，最后用一个softmax
import tensorflow as tf
import numpy as np
def compute_accuraty(v_xs,v_ys):
    global prediction
    y_pre=sess.run(prediction,feed_dict={xs: v_xs, keep_prob:1})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuracy,feed_dict={xs: v_xs, ys: v_ys, keep_prob:1})
    return  result
#手写体数字库（MNIST库)
#数据中包含55000张训练图片，每张图片的分辨率是28×28，所以我们的训练网络输入应该是28×28=784个像素数据。
from tensorflow.examples.tutorials.mnist import input_data
MNIST_data_folder="F:\\code\\python\\git\\flask_study\\TensorFlow-study\\MNIST_data"
mnist = input_data.read_data_sets(MNIST_data_folder, one_hot=True)
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#权重初始化
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
#卷积和池化
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# 第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个4维的Tensor，要求类型为float32和float64其中之一
# 第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维
# 第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
# 第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式（后面会介绍）
# 第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
# 第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape
#
# 第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
#
# 第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
#
# 第四个参数padding：和卷积类似，可以取'VALID' 或者'SAME'
#
# 返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式
xs=tf.placeholder(tf.float32,shape=[None,784])
ys=tf.placeholder(tf.float32,shape=[None,10])

#第一层卷积
#使用5x5x1的大小进行卷积运算，生成32个输出神经元
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(xs, [-1,28,28,1]) #-1表示（变成28*28*1的图片有多少张，patch）
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) #ReLU激活函数，最后进行max pooling
h_pool1 = max_pool_2x2(h_conv1)

#第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
#密集连接层
#经过两次池化28/2/2变成7*7，加入一个有1024个神经元的全连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
#池化层输出的张量reshape成一些向量
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
#输出层
#添加一个softmax层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


#prediction=add_layer(xs,784,10,activation_function=tf.nn.softmax)
#loss funciton
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),axis=1))
#train
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)  # 现在开始train，每次只取100张图片
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob:0.5})
    if i %50 ==0:
        print(compute_accuraty(mnist.test.images,mnist.test.labels))
