#coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#create data
#x_data=np.random.rand(1000).astype(np.float32)
x_data=np.linspace(-1,1,300).astype(np.float32)
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = 2*np.square(x_data) - 0.5 + noise
#y_data=x_data*x_data*0.3+0.3+np.random.random()*10
#搭建模型
Weights=tf.Variable(tf.random_uniform([1],-10.0,10.0))  #tf中变量使用tf.Variable(）来定义。生成的以为向量，大小在[-1,1]之间
#tf.random_uniform([2,3],-1.0,1.0) 则就表示随机生成2x3的矩阵，产生于low和high之间，产生的值是均匀分布的
biases=tf.Variable(tf.zeros([1])) #生成0的向量
y=Weights*x_data*x_data+biases
#计算误差
loss=tf.reduce_mean(tf.square(y-y_data))

#传播误差
optimizer=tf.train.GradientDescentOptimizer(0.5) #0.5表示梯度下降的速率
train=optimizer.minimize(loss)

#训练
#初始化变量，必须先初始化所有之前定义的Variable
init=tf.global_variables_initializer()

sess=tf.Session()
sess.run(init) #初始化非常重要

#尝试可视化绘图
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data,s=3)
plt.ion() #使得程序没有暂停在窗口
plt.show()
for step in range(501):
    sess.run(train)
    if step %20 ==0:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        lines=ax.plot(x_data,sess.run(y),'r',lw=5)
        plt.pause(0.3)
        print(step,sess.run(Weights),sess.run(biases))



