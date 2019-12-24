"""
本脚本将cifar10数据打包成.tfrecord格式的数据

正常情况下我们训练文件夹经常会生成 train, test 或者val文件夹，这些文件夹内部往往会存着成千上万的图片或文本等文件，这些文件被散列存着，这样不仅占用磁盘空间，并且再被一个个读取的时候会非常慢，繁琐。占用大量内存空间（有的大型数据不足以一次性加载）。此时我们TFRecord格式的文件存储形式会很合理的帮我们存储数据。TFRecord内部使用了“Protocol Buffer”二进制数据编码方案，它只占用一个内存块，只需要一次性加载一个二进制文件的方式即可，简单，快速，尤其对大型训练数据很友好。而且当我们的训练数据量比较大的时候，可以将数据分成多个TFRecord文件，来提高处理效率。

"""

import tensorflow as tf
import cv2
import numpy as np
import glob

classification = ['airplane',
                  'automobile',
                  'bird',
                  'cat',
                  'deer',
                  'dog',
                  'frog',
                  'horse',
                  'ship',
                  'truck']


idx = 0
im_data = []  # 所有类别的图片文件
im_labels = []  # 所有文件的标签
for path in classification:  # 对于每个类别
    path = "data/image/train/" + path  # 形如：data/image/train/bird
    im_list = glob.glob(path + "/*")  # 获取这个类别下所有文件
    # Python在方括号中使用for循环，类似[0 for i in range(10)]，叫 列表解析List Comprehensions
    # 根据已有列表，高效创建新列表的方式。
    # 列表解析是Python迭代机制的一种应用，它常用于实现创建新的列表，因此用在[]中。
    # [expression for iter_val in iterable]
    # [expression for iter_val in iterable if cond_expr]
    im_label = [idx for i in range(im_list.__len__())]  # idx就只有0~9
    idx += 1  # 每次换类别加1
    im_data += im_list  # 把本次类别下的图片列表加到全部的图片列表中
    im_labels += im_label  # 把本次的标签加到全部的标签列表中

print(im_labels)
# print(im_data)

# 下面生成.tfrecord文件
tfrecord_file = "data/train.tfrecord"
# 利用TFRecordWriter写入到文件中
writer = tf.compat.v1.python_io.TFRecordWriter(tfrecord_file)

index = [i for i in range(im_data.__len__())]

# 打乱图片的顺序
np.random.shuffle(index)

for i in range(im_data.__len__()):
    im_d = im_data[index[i]]
    im_l = im_labels[index[i]]
    # opencv对图片进行读取
    data = cv2.imread(im_d)
    # 也可以通过这种方式读取图片数据，本身就是byte格式的
    # data = tf.gfile.FastGFile(im_d, "rb").read()
    ex = tf.train.Example(
        # Features是用于描述机器学习模型训练或推理的特征的协议消息，用键值对表示数据。
        # 一个Features中包括可能包含零个或多个值的列表。 这些列表是基本值BytesList，FloatList，Int64List。Feature按名称分类。 Feature的消息包含从名称到功能的映射。
        features=tf.train.Features(
            feature={
                "image": tf.train.Feature(  # 图像数据使用byte类型进行存储
                    bytes_list=tf.train.BytesList(
                        value=[data.tobytes()])),
                "label": tf.train.Feature(  # 标签数据是int类型的数据
                    int64_list=tf.train.Int64List(
                        value=[im_l])),
            }
        )
    )
    # 封装好的数据进行序列化，并且写入tfrecord中
    writer.write(ex.SerializeToString())
# 关闭writer
writer.close()

