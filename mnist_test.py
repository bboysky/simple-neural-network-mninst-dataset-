# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 20:02:24 2017

@author: Administrator
"""


import tensorflow as tf
#load data from mnist website
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
# the correct answer
y_actual = tf.placeholder(tf.float32, shape=[None, 10])

#variable of weight and bias
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#加权变换并进行softmax回归，得到预测概率
y_predict = tf.nn.softmax(tf.matmul(x,w)+b)
#求交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_actual*tf.log(y_predict),1))
#用梯度下降法以0.01的学习率最小化交叉熵使得残差最小
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#在测试阶段，测试准确度计算
correct_prediction = tf.equal(tf.argmax(y_predict,1),tf.argmax(y_actual,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.initialize_all_variables()

with tf.Session() as sess:
        sess.run(init)
        
        for i in range(1000):
            batch_xs,batch_ys = mnist.train.next_batch(100)  #按批次训练，每批100行数据
            sess.run(train_step,feed_dict={x:batch_xs, y_actual:batch_ys})
            if(i%100==0):
                print("accuracy:",sess.run(accuracy,feed_dict={x:mnist.test.images,y_actual:mnist.test.labels})) #每训练100次，测试一次