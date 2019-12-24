# cifar10-tensorflow-read-write
本项目通过Tensorflow对Cifar10数据集进行读写操作。

包括完整代码和详细注释。

## Cifar10数据集介绍
- 由60000个图片组成
- 6万个图片中，5万张用于训练，1万张用于测试
- 每个图片是32x32像素
- 所有图片可以分成10类
- 每个图片都有一个标签，标记属于哪一个类
- 测试集中一个类对应1000张图
- 训练集中将5万张图分为5份
- 类之间的图片是互斥的，不存在类别重叠的情况

## Cifar10数据集分类
![分类](http://aichenwei.oss-ap-southeast-1.aliyuncs.com/github/cifar10.png)

## Cifar10数据集下载
- [官方](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

- [百度网盘](https://pan.baidu.com/s/1AwQUx_KukoScbqlbF_IS-w?_blank) 
    密码：tuqq

## 项目运行需要的环境
- Python 3.7.4

## 项目运行
### convert_cifar10_image.py
这个脚本对数据进行下载，并且转换成图片
首先将文件中下面三行的注释移除
```shell script
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
DATA_DIR = 'data'
download_and_uncompress_tarball(DATA_URL, DATA_DIR)
```
并且执行
```shell script
python3 convert_cifar10_image.py
```
这样脚本会自动下载数据，并且转换成图片。其中训练数据在data/image/train目录下。
如果在线下载比较慢，可以通过百度网盘先将数据集下载到项目根目录下，再执行这个convert_cifar10_image.py脚本，这样会自动跳过从网络上下载。

## tf.train.slice_input_producer()函数
详细说明请访问[《详解tensorflow的tf.train.slice_input_producer文件队列tensor生成器》](https://blog.csdn.net/keyandi/article/details/103683761)

代码参考：reader_cifar10-1.py