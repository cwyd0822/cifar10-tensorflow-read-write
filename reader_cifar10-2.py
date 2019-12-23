import tensorflow as tf

filename = ['data/A.csv', 'data/B.csv', 'data/C.csv']

file_queue = tf.train.string_input_producer(filename,
                                            shuffle=True,
                                            num_epochs=2)
reader = tf.WholeFileReader()
key, value = reader.read(file_queue)

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    tf.train.start_queue_runners(sess=sess)
    for i in range(10):
        print(sess.run([key, value]))