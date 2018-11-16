# TensorFlow 函数注释

## tf.placeholder 
```Python
tf.placeholder(dtype, shape = None, name = None)
```

此函数用于定义过程，在执行的时候再具体赋值.

args:

- dtype : 数据类型. 常用的是 tf.float32, tf.float64 等数值类型

- shape : 数据形状. 默认值是 None, 也就是一维值，也可以是多维值.

- name : 名称

return

- Tensor 类型

## tf.nn.embedding_lookup 函数

tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素. tf.nn.embedding_lookup（tensor, id）:tensor就是输入张量，id就是张量对应的索引，其他的参数不介绍.


## tf.reducemean 函数

```Python
    tf.reduce_mean(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)
```
根据给出的axis在input_tensor上求平均值。除非keep_dims为真，axis中的每个的张量秩会减少1。如果keep_dims为真，求平均值的维度的长度都会保持为1.如果不设置axis，所有维度上的元素都会被求平均值，并且只会返回一个只有一个元素的张量。

## random.sample 函数

```Python
random.sample(seq,n)
```
从序列 seq 中随机选出 n 个独立且随机的元素

## tf.square 函数

```Python
tf.square() 函数
```

求每个元素的平方

## yeild 关键词

和 return 一致，但是返回的是迭代器

## numpy.array 用法

```Python
x = numpy.array([1,2,3,4])
x[:,None]
#[[1],
# [2],
# [3],
# [4]]
# ：表示为全部为行 ，None 表示列没有
```