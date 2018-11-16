import tensorflow as tf

# declaim w1 and w2. seed is 随机数种子
# 保证运行的结果是一样的

w1 = tf.Variable(tf.random_normal([2,3],stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1, seed=1))

# 将 x 定义为一个常量
# x = tf.constant([[0.7,0.9]])

# 将 x 定义为一个 placeholder
x = tf.placeholder(tf.float32, shape=(3, 2), name="input")

# 隐含层输出
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()

# 方法 1
# sess.run(w1.initializer)
# sess.run(w2.initializer)

# 方法 2
init_op = tf.global_variables_initializer()
sess.run(init_op)

# print(sess.run(y))
# [[3.957578]]
print(sess.run(y, feed_dict={x:[[0.7,0.9],[0.1,0.4],[0.5,0.8]]}))

sess.close()