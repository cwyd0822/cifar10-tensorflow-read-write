#!/usr/bin/python
# coding:utf-8
import tensorflow as tf
import numpy as np

images = np.random.random([5, 2])  # 5x2的矩阵
print(images)
label = np.asarray(range(0, 5))  # [0, 1, 2, 3, 4]
print(label)
# 将数组转换为张量
images = tf.cast(images, tf.float32)
print(images)
label = tf.cast(label, tf.int32)
print(label)
# 切片
input_queue = tf.train.slice_input_producer([images, label], shuffle=False)
# 按顺序读取队列中的数据
image_batch, label_batch = tf.train.batch(input_queue, batch_size=10, num_threads=1, capacity=64)

with tf.Session() as sess:
    # 线程的协调器
    coord = tf.train.Coordinator()
    # 开始在图表中收集队列运行器
    threads = tf.train.start_queue_runners(sess, coord)
    image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
    for j in range(5):
        print(image_batch_v[j]),
        print(label_batch_v[j])
    # 请求线程结束
    coord.request_stop()
    # 等待线程终止
    coord.join(threads)

