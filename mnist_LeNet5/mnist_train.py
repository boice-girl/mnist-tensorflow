# -*- coding:utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.2         # 基础的学习率
LEARNING_RATE_DECAY = 0.99       # 学习率的衰减率
REGULARIZATION_RATE = 0.0001     # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 50000          # 训练轮数
MOVING_AVERAGE_DECAY = 0.99      # 滑动平均衰减率

MODEL_SAVE_PATH = "./model"      # 模型保存的路径与名称
MODEL_NAME = "model.ckpt"

def train(mnist):
    x = tf.placeholder(tf.float32,                      #shape是一个四维的矩阵，第一维代表一个batch的样本数
                       [BATCH_SIZE,
                        mnist_inference.IMAGE_SIZE,     #第二维跟第三维代表图片的大小
                        mnist_inference.IMAGE_SIZE,
                        mnist_inference.NUM_CHANNELS],  #第四维代表图片的通道数
                       name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.NUM_LABELS], name='y-input')
    
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算前向传播的结果
    y = mnist_inference.inference(x, 1, regularizer)
    # 定义训练轮数的参数,设置为不可训练的
    global_step = tf.Variable(0, trainable=False)
    # 初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # 在代表神经网络的参数上使用滑动平均
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # 计算交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 计算模型的正则化损失,一般只计算神经网络边上权重的正则化损失
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    # 使用此算法优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')
    
    # 初始化TensorFlow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                        mnist_inference.IMAGE_SIZE,
                        mnist_inference.IMAGE_SIZE,
                        mnist_inference.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
            if i % 1000 == 0:
                print('After %d training steps, loss on training batch is %g' % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),global_step=global_step)

def main(argv=None):
    mnist = input_data.read_data_sets('./data', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
