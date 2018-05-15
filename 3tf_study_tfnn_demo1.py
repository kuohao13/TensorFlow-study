#coding:utf-8
import tensorflow as tf
import numpy as np
#1层10个神经元的拟合一元二次函数的简单列子
# 输入层1个、隐藏层10个、输出层1个的神经网络
def add_layer(input_data,in_size,out_size,activation_function=None):
    # input_data:输入的数据 in_size 输入的数据的特征维度，out_size输出的维度（第一层神经元可以认为神经元个数）
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))#随机初始化权重
    biase=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(input_data,Weights)+biase
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs

#数据源 随机造一组数据
x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis] #300x1
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

#占位符
xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])
# 这里的None代表无论输入有多少都可以，因为输入只有一个特征，所以这里是1

#搭建网络
l1=add_layer(xs,1,10,activation_function=tf.nn.relu)
prediction=add_layer(l1,10,1,activation_function=None)

#损失函数 loss_function
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                                  reduction_indices=[1]))
#reduction_indices=[1] 表示通过行累加或均值，现在更多用axis=[1]

#优化损失函数
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#采用梯度下降的一种方法，0.1表示是学习速率（步长）

#变量初始化
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

#训练
for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i% 50 ==0:
        # 每50步我们输出一下机器学习的误差
        print("now,run in:%s,loss is %s"%(i,sess.run(loss,feed_dict={xs:x_data,ys:y_data})))


