import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="MNIST_model/"
MODEL_NAME="mnist_model"

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', regularizer(weights))
    return weights

def inference(input_tensor, avg_class, regularizer, reuse):

    if avg_class == None:
        with tf.variable_scope('layer1', reuse=reuse):

            weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
            biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
            layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

        with tf.variable_scope('layer2', reuse=reuse):
            weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
            biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
            layer2 = tf.matmul(layer1, weights) + biases
        return layer2

    else:
        with tf.variable_scope('layer1', reuse=reuse):

            weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
            biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
            layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights)) + biases)


        with tf.variable_scope('layer2', reuse=reuse):
            weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
            biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
            layer2 = tf.matmul(layer1, avg_class.average(weights)) + biases

        return layer2

def train(mnist):

    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    
    global_step = tf.Variable(0, trainable=False)
    
    
    y = inference(x,None, regularizer, False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 权值共享. 保证和  y = inference(x,None, regularizer, False) 使用相同的权值矩阵
    average_y = inference(x, variable_averages, regularizer, True)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                test_acc = sess.run(accuracy, feed_dict=test_feed)
                print("After %d training step(s), validation accuracy using average model is %g , test accuracy using average model is %g" % (i, validate_acc, test_acc))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("D:/MNIST/", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
