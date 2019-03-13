# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans
# deal_data是我本地的样本处理脚本，这里就不放上来了
from deal_data import train_X, train_Y, test_X, test_Y
# train_X：训练集图片像素矩阵，【760*780】 760张图片
# train_Y： 训练集图片的标签，onehot格式【760*24】
# test_X，test_Y 为测试数据，格式和训练集一致

# 定义占位符
TRAIN_X = tf.placeholder("float", [None, 780])
TEST_X = tf.placeholder("float", [780])

# 计算L1正则距离
# tf.abs(tf.add(train_X, tf.negative(test_X)))的计算结果是一个（760*780）矩阵，表示测试图片与每一张训练集图片的像素差值
# distance的结果为一个（760，）向量，表示测试图片与每一张训练集图片像素总差值
distance = tf.reduce_sum(tf.abs(tf.add(TRAIN_X, tf.negative(TEST_X))), reduction_indices=1)
# 预测结果pred取距离最小的那个训练集图片的索引
pred = tf.arg_min(distance, 0)

accuracy = 0.

# 初始化变量
init = tf.initialize_all_variables()

# 开始迭代
if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(init)

        # 遍历测试集
        for i in range(len(test_X)):
            # 获取最近邻
            nearest_idx = sess.run(pred, feed_dict={TRAIN_X: train_X, TEST_X: test_X[i, :]})
            # 如果最近邻标签索引和真是标签索引一致，则预测准确
            if np.argmax(train_Y[nearest_idx]) == np.argmax(test_Y[i]):
                accuracy += 1. / len(test_X)
        print("Done!")
        print("Accuracy:", accuracy)
