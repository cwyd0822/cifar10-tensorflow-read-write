import tensorflow as tf
import cv2
import numpy as np
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

import glob
idx = 0
im_data = []
im_labels = []
for path in classification:
    path = "data/image/test/" + path
    im_list = glob.glob(path + "/*")
    im_label = [idx for i in  range(im_list.__len__())]
    idx += 1
    im_data += im_list
    im_labels += im_label

print(im_labels)
print(im_data)

tfrecord_file = "data/test.tfrecord"
writer = tf.python_io.TFRecordWriter(tfrecord_file)

index = [i for i in range(im_data.__len__())]

np.random.shuffle(index)

for i in range(im_data.__len__()):
    im_d = im_data[index[i]]
    im_l = im_labels[index[i]]
    data = cv2.imread(im_d)
    #data = tf.gfile.FastGFile(im_d, "rb").read()
    ex = tf.train.Example(
        features = tf.train.Features(
            feature = {
                "image":tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[data.tobytes()])),
                "label": tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=[im_l])),
            }
        )
    )
    writer.write(ex.SerializeToString())

writer.close()
