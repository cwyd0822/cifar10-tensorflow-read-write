"""
本脚本对cifar10数据进行解析，转换成图片，生成训练图片和测试图片。
"""

import urllib.request
import os
import sys
import tarfile
import glob
import pickle
import numpy as np
import cv2


# 通过这个函数完成对数据集的下载和解压
# tarball_url 表示cifar10数据集的下载链接
# dataset_dir 表示存储的路径

# 执行下面的代码可以完成数据集的下载和解压
# DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
# DATA_DIR = 'data'

# download_and_uncompress_tarball(DATA_URL, DATA_DIR)
def download_and_uncompress_tarball(tarball_url, dataset_dir):
    """Downloads the `tarball_url` and uncompresses it locally.
  Args:
    tarball_url: The URL of a tarball file.
    dataset_dir: The directory where the temporary files are stored.
  """
    # tarball_url='http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    filename = tarball_url.split('/')[-1]  # 文件名，通过/拆分字符串，取最后一节，也就是cifar-10-python.tar.gz
    # dataset_dir = 'data'
    # os.path.join()路径拼接，/data/cifar-10-python.tar.gz
    filepath = os.path.join(dataset_dir, filename)

    # 定义进度函数，分块下载
    # count 第几个块
    # block_size 每个块的大小
    # total_size 总的大小
    def _progress(count, block_size, total_size):
        # 等价于print() print底层调用的就是sys.stdout.write()
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    # 如果文件还不存在，才下载，否则不下载
    if not os.path.isfile(filepath):
        # urlretrieve(url, filename=None, reporthook=None, data=None)
        # 参数url：下载链接地址
        # 参数filename：指定了保存本地路径（如果参数未指定，urllib会生成一个临时文件保存数据。）
        # 参数reporthook：是一个回调函数，当连接上服务器、以及相应的数据块传输完毕时会触发该回调，我们可以利用这个回调函数来显示当前的下载进度。
        # 参数data：指post导服务器的数据，该方法返回一个包含两个元素的(filename, headers)元组，filename表示保存到本地的路径，header表示服务器的响应头
        filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
        print()

    else:
        print('File already existed!')

    # 获取文件的信息
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    # 将gz文件解压到dataset_dir指定的文件夹
    tarfile.open(filepath, 'r:gz').extractall(dataset_dir)


# 预定义10个分类
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


# pickle用于序列化
# 用于python特有的类型和python的数据类型间进行转换
# pickle提供四个功能：dumps,dump,loads,load
# pickle可以存储的数据类型
# - 所有python支持的原生类型：布尔值，整数，浮点数，复数，字符串，字节，None。
# - 由任何原生类型组成的列表，元组，字典和集合。
# - 函数，类，类的实例
def unpickle(file):
    with open(file, 'rb') as fo:  # 打开文件
        dict = pickle.load(fo, encoding='bytes')  # 是以字典的方式序列化数据
    return dict


# DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
# DATA_DIR = 'data'

# download_and_uncompress_tarball(DATA_URL, DATA_DIR)

# 选择cifar10的路径，用相对路径
folders = 'data/cifar-10-batches-py'

# 获取目录下所有匹配到的训练集文件
trfiles = glob.glob(folders + "/data_batch*")

data = []  # 二进制的数据
labels = []  # 标签列表
for file in trfiles:  # 对于每个文件
    dt = unpickle(file)  # 得到反序列化后的数据 {"data": [byte], "labels": [int8]}
    data += list(dt[b"data"])  # 转化为list
    labels += list(dt[b"labels"])

# labels形如[1, 2, 3, 4, 6, ...]
# data 形如：... array([163, 173, 158, ..., 101, 100,  95], dtype=uint8) ...
print(labels)
print(len(data))

# 3通道32*32的图像数据
# [-1, 3, 32, 32]表示将data重新整理成3*32*32的图片，-1表示转化的数量根据实际情况确定
# numpy.reshape(a, newshape)
# a : 数组——需要处理的数据。
# 新的格式——整数或整数数组，如(2,3)表示2行3列。新的形状应该与原来的形状兼容，即行数和列数相乘后等于a中元素的数量。如果是整数，则结果将是长度的一维数组，所以这个整数必须等于a中元素数量。若这里是一个整数数组，那么其中一个数据可以为-1。在这种情况下，这个个值python会自动从根据第二个数值和剩余维度推断出来。
imgs = np.reshape(data, [-1, 3, 32, 32])

# 遍历所有图片
# imgs.shape[0]表示图片的数量
for i in range(imgs.shape[0]):
    im_data = imgs[i, ...]
    im_data = np.transpose(im_data, [1, 2, 0])  # 修改通道顺序，把通道移动到最后[32, 32, 3]
    im_data = cv2.cvtColor(im_data, cv2.COLOR_RGB2BGR)  # 修改图像通道为BGR

    # 把图像数据写成文件: data/image/test/airplane
    f = "{}/{}".format("data/image/train", classification[labels[i]])

    # 确保文件夹存在
    if not os.path.exists(f):
        os.mkdir(f)

    # 文件名形如：data/image/train/airplane/1.jpg
    cv2.imwrite("{}/{}.jpg".format(f, str(i)), im_data)
