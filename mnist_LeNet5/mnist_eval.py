# -*- coding: utf-8 -*-
import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train

EVAL_INTERVAL_SECS = 10
def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,  # shape是一个四维的矩阵，第一维代表一个batch的样本数
                           [5000,
                            mnist_inference.IMAGE_SIZE,  # 第二维跟第三维代表图片的大小
                            mnist_inference.IMAGE_SIZE,
                            mnist_inference.NUM_CHANNELS],  # 第四维代表图片的通道数
                           name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.NUM_LABELS], name='y-input')
        
        reshaped_x = np.reshape(mnist.validation.images, (5000,
                                      mnist_inference.IMAGE_SIZE,
                                      mnist_inference.IMAGE_SIZE,
                                      mnist_inference.NUM_CHANNELS))
        validate_feed = {
            x: reshaped_x,
            y_: mnist.validation.labels
        }
        y = mnist_inference.inference(x, 0, None)
        #计算正确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # 通过变量重命名的方式加载模型,这样就不用调用滑动平均的函数来获取平均值了
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                print ckpt.model_checkpoint_path
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print('After %s training steps, validation accuracy = %g' % (global_step, accuracy_score))
                else:
                    print ('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)
def main(argv=None):
    mnist = input_data.read_data_sets('./data', one_hot=True)
    evaluate(mnist)
if __name__ == '__main__':
    tf.app.run()