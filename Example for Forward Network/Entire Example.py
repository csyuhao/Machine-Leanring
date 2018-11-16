import tensorflow as tf
from numpy.random import RandomState

# define batch size
batch_size = 8

# define weight parameters
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

# on one dimension of the 'shape' can use easily different batch size.
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')  

# define the process of forward propagation
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# define the loss function and back propagation 
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
)

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# produce a simulative dataset
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

# create session
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print(sess.run(w1)) #[[-0.81131822  1.48459876  0.06532937], [-2.4427042   0.0992484   0.59122431]]
    print(sess.run(w2)) #[[-0.81131822], [ 1.48459876], [ 0.06532937]]

    STEPS = 5000
    for i in range(STEPS):
        # choose the batch size number of dataset
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)

        # update weights
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

        if i % 1000 == 0:
            # calculate the cross entropy
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x:X, y_:Y})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))
    
    print(sess.run(w1)) # [[-1.96182752  2.58235407  1.68203771], [-3.46817183  1.06982315  2.11788988]]
    print(sess.run(w2)) # [[-1.82471502], [ 2.68546653], [ 1.41819501]]

