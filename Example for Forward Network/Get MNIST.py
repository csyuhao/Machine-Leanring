from tensorflow.examples.tutorials.mnist import input_data
# 载入MNIST数据集，如果指定地址/path/to/MNIST_data下没有已经下载好的数据，
# 那么TensorFlow会自动从表5-1给出的网址下载数据。
mnist = input_data.read_data_sets("D:/MNIST/", one_hot=True)
# 打印Training data size: 55000。
print("Training data size: %d" % (mnist.train.num_examples))
# 打印Validating data size: 5000。
print("Validation data size:%d" % (mnist.validation.num_examples))
# 打印Testing data size: 10000.
print("Testing data size:%d" % (mnist.test.num_examples))
# 打印Example training data: [ 0. 0. 0. … 0.380 0.376 … 0. ]。
print("Example training data: %s" % (mnist.train.images[0]))
# 打印Example training data label:
# [ 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
print("Example training data label: %s" % (mnist.train.labels[0]))