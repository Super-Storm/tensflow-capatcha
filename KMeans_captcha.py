# -*- coding:utf-8 -*-
# 使用KMeans实现验证码识别
#   Author  :   fx
#   E-mail  :   pythonist@126.com

import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans
# deal_data是我本地的样本处理脚本，这里就不放上来了
from deal_data import train_X, train_Y, test_X, test_Y
train_X = train_X / 255
train_Y = train_Y / 255
test_X = test_X / 255
test_Y = test_Y / 255
# 这里X是一个？*780的数组，对应？张图片。每张图片大小为30*26，拉成一维后：780
# Y 是每张图片对应的分类，值为onehot形式，维度为（?,24）

# 参数设置
num_steps = 1000   # 迭代次数
k = 24  # 聚类中心数目
num_classes = 24    # 数据标签数目

# 定义占位符
X = tf.placeholder(dtype=tf.float32, shape=[None, 780])
Y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])

# kmeans初始化
# 使用余弦相似度，cosine
kmeans = KMeans(inputs=X, num_clusters=k, distance_metric="cosine", use_mini_batch=True)
all_scores, cluster_idx, _, _, init_op, training_op = kmeans.training_graph()
# all_scores： 所有的输入与对应对的中心的相似度，维度为（?，24）
# cluster_idx：每个图片对应的聚类中心点,
# init_op：kmeans聚类中心的初始化操作节点
# training_op：kmeans的训练迭代节点

cluster_index = cluster_idx[0]  # cluster_idx是一个元组，第一个元素才是每个图片对应的聚类中心点索引
avg_distance = tf.reduce_mean(all_scores)   # 计算相似度的平均值
init_param = tf.global_variables_initializer()  # tensorflow全部变量的初始化操作节点
with tf.Session() as sess:
    # 运行tensorflow变量初始化
    sess.run(init_param)
    # 运行kmeans初始化
    sess.run(init_op, feed_dict={X: train_X})
    # 开始迭代
    for i in range(1, num_steps+1):
        _, distance, idx = sess.run([training_op, avg_distance, cluster_index], feed_dict={X: train_X})
        if i % 10 == 0 or i == 1:
            print("Step: %d, distance=%f" % (i, distance))

    counts = np.zeros(shape=(k, num_classes))
    for i in range(len(idx)):
        # counts 记录的时每一个聚类中心下的图片分类统计，
        counts[idx[i]] += train_Y[i]    # 维度：（len(idx)，24）
    # clustags_to_imtags记录的是每一个聚类中心下分布最多的图片分类，
    clustags_to_imtags = [np.argmax(c) for c in counts]     # 维度：（len(idx)）
    # 将clustags_to_imtags转化成张量
    clustags_to_imtags = tf.convert_to_tensor(clustags_to_imtags)

    # ------------------------------------------------
    # 下面是验证时的运算图
    # pred_label是将每一个图片的聚类中心的分布最多的那个图片分类作为预测的结果
    pred_label = tf.nn.embedding_lookup(clustags_to_imtags, cluster_index)  # 维度：（len(idx)）
    # 计算正确率
    equal_prediction = tf.equal(pred_label, tf.cast(tf.argmax(Y, 1), tf.int32))
    accuracy_op = tf.reduce_mean(tf.cast(equal_prediction, tf.float32))  # tf.cast(equal_prediction, tf.float32) 可以将bool值转化为1.0或者0.0,这样计算平均值接可以得到正确率

    # 测试模型
    test_x, test_y = test_X, test_Y
    print("验证集准确率:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))
