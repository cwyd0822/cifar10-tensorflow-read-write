#!/usr/bin/python
# coding:utf-8
import tensorflow as tf
import numpy as np

images = np.random.random([5, 2])
label = np.asarray(range(0, 5))
# 列表转成张量
images = tf.cast(images, tf.float32)
label = tf.cast(label, tf.int32)
# 输入张量列表
input_queue = tf.train.slice_input_producer([images, label], shuffle=False)
# 将队列中数据打乱后再读取出来
image_batch, label_batch = tf.train.shuffle_batch(input_queue, batch_size=10, num_threads=1, capacity=64, min_after_dequeue=1)

with tf.Session() as sess:
    # 线程的协调器
    coord = tf.train.Coordinator()
    # 开始在图表中收集队列运行器
    threads = tf.train.start_queue_runners(sess, coord)
    image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
    for j in range(5):
        # print(image_batch_v.shape, label_batch_v[j])
        print(image_batch_v[j]),
        print(label_batch_v[j])
    # 请求线程结束
    coord.request_stop()
    # 等待线程终止
    coord.join(threads)
