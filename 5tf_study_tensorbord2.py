#coding:utf-8
import tensorflow as tf
import numpy as np
#添加了,n_layer参数
def add_layer(inputs, in_size, out_size,n_layer, activation_function=None):
    # add one more layer and return the output of this layer
    layer_name="layer%s"%n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            #添加weights的histogram
            tf.summary.histogram(layer_name+'/weights',Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            #添加biases的histogram
            tf.summary.histogram(layer_name+'/biases',biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        #添加outputs的histogram
        tf.summary.histogram(layer_name+'/outpus',outputs)
        return outputs


#随机生成数据
x_data=np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data=np.square(x_data)-0.5+noise

#输入数据
with tf.name_scope("inputs"):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='x_input')

#添加神经层数
l1=add_layer(xs,1,10,n_layer=1,activation_function=tf.nn.relu)
prediction=add_layer(l1,10,1,n_layer=2,activation_function=None)

#loss function
with tf.name_scope("loss"):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(prediction-ys),axis=1))
    #添加loss function的event
    tf.summary.scalar('loss',loss)

#train
with tf.name_scope("train"):
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess=tf.Session()
#合并所有的histogram
merge=tf.summary.merge_all()

write=tf.summary.FileWriter("logs",sess.graph)

init=tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    # result = sess.run(merge, feed_dict={xs: x_data, ys: y_data})
    if i%50 ==0:
        result=sess.run(merge,feed_dict={xs:x_data,ys:y_data})
        write.add_summary(result,i)

