#coding:utf-8
import tensorflow as tf
import numpy as np
#create data
x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3
#搭建模型
Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))  #tf中变量使用tf.Variable(）来定义。生成的以为向量，大小在[-1,1]之间
#tf.random_uniform([2,3],-1.0,1.0) 则就表示随机生成2x3的矩阵，产生于low和high之间，产生的值是均匀分布的
biases=tf.Variable(tf.zeros([1])) #生成0的向量
