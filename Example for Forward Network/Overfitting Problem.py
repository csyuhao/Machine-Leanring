import tensorflow as tf

def get_weight(shape, lamb):
    # define a new variable
    var = tf.Variable(tf.random_normal(shape), dtype = tf.float32)
    # add_to_collection : put the l2 regularizer term of 'var' int collecton
    # the first parameter is the name of collection is 'losses'
    # the second parameter is the value which be added into collection
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lamb)(var))

    # return value
    return var

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
batch_size = 8
# define the number of each layer's neuron
layer_dimension = [2, 10, 10, 10, 1]
# the number of layers
n_layers = len(layer_dimension)
# this parameter is the deepest layer in forward propagation. 
# at begin, it is input layer 
cur_layer = x
# the number of current layer's nodes
in_dimension = layer_dimension[0]
# creating the 5 layers network in a circulation
for i in range(1, n_layers):
    # layer_dimension[i] is the numeber of the next layer's node
    out_dimension = layer_dimension[i]
    # create the weight variable of current layer, and add the l2 regularizer of this to collection
    weight = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    # use ReLU activation function
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bais)
    # update 'in_dimension'
    in_dimension = out_dimension

# loss in training dataset
mse_loss = tf.reduce_mean(tf.square(y - cur_layer))
tf.add_to_collection('losses', mse_loss)
# get_collections to get a collection
loss = tf.add_n(tf.get_collection('losses')) 