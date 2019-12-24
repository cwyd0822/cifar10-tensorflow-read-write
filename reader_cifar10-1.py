import tensorflow as tf

# 定义4个图片路径列表
images = ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg']
# 定义4个Label的列表
labels = [1, 2, 3, 4]

# tf.train.slice_input_producer是一个tensor生成器，作用是按照设定，每次从一个tensor列表中按顺序或者随机抽取出一个tensor放入文件名队列。 slice_input_producer(
# tensor_list, num_epochs=None, shuffle=True, seed=None, capacity=32, shared_name=None, name=None) 第一个参数
# tensor_list：包含一系列tensor的列表，表中tensor的第一维度的值必须相等，即个数必须相等，有多少个图像，就应该有多少个对应的标签。 第二个参数num_epochs:
# 可选参数，是一个整数值，代表迭代的次数，如果设置 num_epochs=None,生成器可以无限次遍历tensor列表，如果设置为 num_epochs=N，生成器只能遍历tensor列表N次。 第三个参数shuffle：
# bool类型，设置是否打乱样本的顺序。一般情况下，如果shuffle=True，生成的样本顺序就被打乱了，在批处理的时候不需要再次打乱样本，使用 tf.train.batch函数就可以了;如果shuffle=False,
# 就需要在批处理时候使用 tf.train.shuffle_batch函数打乱样本。
# 第四个参数seed: 可选的整数，是生成随机数的种子，在第三个参数设置为shuffle=True的情况下才有用。
# 第五个参数capacity：设置tensor列表的容量。
# 第六个参数shared_name：可选参数，如果设置一个‘shared_name’，则在不同的上下文环境（Session）中可以通过这个名字共享生成的tensor。
# 第七个参数name：可选，设置操作的名称。
[images, labels] = tf.train.slice_input_producer([images, labels],
                              num_epochs=2,
                              shuffle=True)

with tf.Session() as sess:
    # 对全局的变量进行初始化
    sess.run(tf.local_variables_initializer())

    # TensorFlow的Session对象是支持多线程的，可以在同一个会话（Session）中创建多个线程，并行执行。在Session中的所有线程都必须能被同步终止，异常必须能被正确捕获并报告，会话终止的时候，
    # 队列必须能被正确地关闭。 TensorFlow提供了两个类来实现对Session中多线程的管理：tf.Coordinator和 tf.QueueRunner，这两个类往往一起使用。
    # Coordinator类用来管理在Session中的多个线程，可以用来同时停止多个工作线程并且向那个在等待所有工作线程终止的程序报告异常，该线程捕获到这个异常之后就会终止所有线程。使用
    # tf.train.Coordinator()来创建一个线程管理器（协调器）对象。
    # QueueRunner类用来启动tensor的入队线程，可以用来启动多个工作线程同时将多个tensor（训练数据）推送入文件名称队列中，具体执行函数是 tf.train.start_queue_runners ， 只有调用
    # tf.train.start_queue_runners 之后，才会真正把tensor推入内存序列中，供计算单元调用，否则会由于内存序列为空，数据流图会处于一直等待状态。
    tf.train.start_queue_runners(sess=sess)  # 启动队列填充的线程

    for i in range(8):  # 从文件队列中获取数据， 4 * 2epoch
        print(sess.run([images, labels]))
