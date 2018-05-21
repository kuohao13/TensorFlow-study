#coding:utf-8
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

#load data
digits =load_digits()
X=digits.data
y=digits.target
y=LabelBinarizer().fit_transform(y)#将标签二值化LabelBinarizer
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=.3) #随机划分训练集和测试集 .test_size=.3表示验证集占训练集30%

def add_layer(inputs,in_size,out_size,layer_name,activation_function=None,):
    Weighs=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1,)
    Wx_plus_b=tf.matmul(inputs,Weighs)+biases
    #在这里droup out
    Wx_plus_b=tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b,)
    tf.summary.histogram(layer_name+'/outpus',outputs)
    return outputs


# define placeholder for inputs to network
keep_prob=tf.placeholder(tf.float32)
xs=tf.placeholder(tf.float32,[None,64])#8x8
ys=tf.placeholder(tf.float32,[None,10])

#add output layer
l1=add_layer(xs,64,50,'L1',activation_function=tf.nn.tanh)
prediction=add_layer(l1,50,10,'L2',activation_function=tf.nn.softmax)

cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),axis=1)) #loss
tf.summary.scalar('loss',cross_entropy)
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess=tf.Session()
merged=tf.summary.merge_all()


train_writer=tf.summary.FileWriter("logs/train",sess.graph)
test_writer=tf.summary.FileWriter("logs/test",sess.graph)

init=tf.global_variables_initializer()
sess.run(init)
for i in range(500):
    #通过keep_prob:0.5决定不要保留概率，即我们要保留的结果所占比例，当keep_prob=1的时候，相当于100%保留，也就是dropout没有起作用
    sess.run(train_step,feed_dict={xs:x_train,ys:y_train,keep_prob:1})
    if i % 50==0:
        # record loss
        train_result = sess.run(merged, feed_dict={xs: x_train, ys: y_train, keep_prob: 1})
        test_result = sess.run(merged, feed_dict={xs: x_test, ys: y_test, keep_prob: 1})
        train_writer.add_summary(train_result,i)
        test_writer.add_summary(test_result,i)

