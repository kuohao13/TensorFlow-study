#coding:utf-8
import tensorflow as tf
import numpy as np
state=tf.Variable(0,name='counter')
# 定义常量 one
one=tf.constant(1)
# 定义加法步骤 (注: 此步并没有直接计算)
new_value=tf.add(state,one)
# 将 State 更新成 new_value
update=tf.assign(state,new_value)

init=tf.global_variables_initializer()

#使用session
with tf.Session() as sess:
    sess.run(init)
    for _ in range(10):
        result=sess.run(update)
        print('state is value:%s'% result)


#placeholder的使用
#Tensorflow 如果想要从外部传入data, 那就需要用到 tf.placeholder(),
        # 然后以这种形式传输数据 sess.run(***, feed_dict={input: **}).

input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)

#做乘积运算
output=tf.multiply(input1,input2)

with tf.Session() as sess:
    result=sess.run(output,feed_dict={input1:3,input2:4})
    print(result)