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


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
# DATA_URL = 'http://aichenwei.oss-ap-southeast-1.aliyuncs.com/github/cifar-10-python.tar.gz'
# DATA_DIR = 'data'

# download_and_uncompress_tarball(DATA_URL, DATA_DIR)

'''
folders = '/home/aiserver/muke/data_manager/data/cifar-10-batches-py'

trfiles = glob.glob(folders + "/test_batch*")

data  = []
labels = []
for file in trfiles:
    dt = unpickle(file)
    data += list(dt[b"data"])
    labels += list(dt[b"labels"])

print(labels)

imgs = np.reshape(data, [-1, 3, 32, 32])

for i in range(imgs.shape[0]):
    im_data = imgs[i, ...]
    im_data = np.transpose(im_data, [1, 2, 0])
    im_data = cv2.cvtColor(im_data, cv2.COLOR_RGB2BGR)

    f = "{}/{}".format("data/image/test", classification[labels[i]])

    if not os.path.exists(f):
        os.mkdir(f)

    cv2.imwrite("{}/{}.jpg".format(f, str(i)), im_data)
'''
