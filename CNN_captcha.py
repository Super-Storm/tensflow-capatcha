# -*- coding:utf-8 -*-
# 使用CNN实现验证码识别
#   Author  :   fx
#   E-mail  :   pythonist@126.com
import numpy as np
import tensorflow as tf
from utils import read_data

train_dir = "data"
test_dir = "test_data"
# train标志着是训练还是测试
train = False
# 模型最后的保存路径
model_path = "model/image_model"

# 这是样本的标签种类
char_to_digit = ["零","壹","贰","叁","肆","伍","陆","柒","捌","玖","拾","一","二","三","四","五","六","七","八","九","加","减","乘","除"]

fpaths, datas, labels = read_data(train_dir)
test_fpath, test_datas, test_labels = read_data(test_dir)
data_len = datas.shape[0]

# n_classes 表示有多少类图片
n_classes = len(set(labels))


# 定义占位符，存放图片和对应的标签 图片数据大小为30*26*1，存放的数据为像素值归一化后的值
X = tf.placeholder(tf.float32, [None, 30, 26, 1])
Y = tf.placeholder(tf.int32, [None])

# drop为dropout参数，为一个百分比，表示反向传播时，选取一部分参数不进行更新，减少过拟合，训练时为0.25，测试时为0
drop = tf.placeholder(tf.float32)

# 定义第一层卷积，20个卷积核，核大小为1*1，即全卷积，relu激活
conv1 = tf.layers.conv2d(X, 20, 1, activation=tf.nn.relu)

# 定义第二层卷积, 20个卷积核, 核大小为1*1，Relu激活
conv2 = tf.layers.conv2d(conv1, 20, 1, activation=tf.nn.relu)


# 将三维向量拉伸为一维
flat = tf.layers.flatten(conv2)

# 全连接，将输入转换成一个1000维向量，还是采用relu激活
fc = tf.layers.dense(flat, 1000, activation=tf.nn.relu)

# 计算dropout
drop_func = tf.layers.dropout(fc, drop)

# 这里再次全连阶，压缩到与分类维度对应的向量
logits = tf.layers.dense(drop_func, n_classes)

# tf.argmax返回的是指定维度上最大值的索引值index
# 这里pred_labels可以用来标志最后的预测结果
pred_labels = tf.argmax(logits, 1)


# 损失函数采用交叉熵，
loss = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(Y, n_classes),
    logits=logits
)
# softmax_cross_entropy_with_logits返回的不是一个具体的值，而是一个向量，这里需要求平均值
m_loss = tf.reduce_mean(loss)

# 定义优化器，学习率为0.001
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(m_loss)


# saver用来保存训练的模型
saver = tf.train.Saver()
if __name__ == '__main__':
    with tf.Session() as sess:

        if train:
            print("train")
            # init
            sess.run(tf.global_variables_initializer())
            # 迭代50次
            for i in range(100):
                _, loss_v = sess.run([optimizer, m_loss], feed_dict={
                                                                    X: datas,
                                                                    Y: labels,
                                                                    drop: 0,
                                                                    }
                                     )
                if i % 10 == 0:
                    print("step:{}-->loss:{}".format(i, loss_v))
            saver.save(sess, model_path)
            print("Done!，save as :{}".format(model_path))
        else:
            # 测试
            print("test")
            saver.restore(sess, model_path)
            print("recover from:{}".format(model_path))
            # label_map是模型输出值与实际分类标签的分类
            label_map = {k: v for k, v in enumerate(char_to_digit)}
            pred_val = sess.run(pred_labels, feed_dict={
                X: test_datas,
                Y: test_labels,
                drop: 0
            })
            err_count = 0
            for fpath, real_label, predicted_label in zip(test_fpath, test_labels, pred_val):
                # 将预测的标签索引值转换为对应分类
                real_label_name = label_map[real_label]
                pred_name = label_map[predicted_label]
                if real_label_name != pred_name:
                    err_count += 1
            print(1 - err_count/len(test_datas))

