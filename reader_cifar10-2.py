"""
读取文件数据
"""

import tensorflow as tf

# 我们放了3个文件在相应的位置
filename = ['data/A.csv', 'data/B.csv', 'data/C.csv']

# 将文件的路径作为参数传入函数
# 输出是文件队列，无法直接获取文件的值
file_queue = tf.train.string_input_producer(filename,
                                            shuffle=True,
                                            num_epochs=2)

# 文件读取器
reader = tf.WholeFileReader()
# key：文件名 value:文件值
key, value = reader.read(file_queue)

with tf.Session() as sess:
    # 对局部变量进行赋值
    sess.run(tf.local_variables_initializer())
    tf.train.start_queue_runners(sess=sess)  # 定义文件队列填充的线程
    for i in range(6):  # 文件数量3 * 2epochs
        print(sess.run([key, value]))


