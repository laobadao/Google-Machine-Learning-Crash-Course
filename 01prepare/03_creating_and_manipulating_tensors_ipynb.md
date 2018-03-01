
#### Copyright 2017 Google LLC.


```python
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

 # 创建和操控张量

 **学习目标：**
  * 初始化 TensorFlow `变量`并赋值
  * 创建和操控张量
  * 回忆线性代数中的加法和乘法知识（如果这些内容对您来说很陌生，请参阅矩阵[加法](https://en.wikipedia.org/wiki/Matrix_addition)和[乘法](https://en.wikipedia.org/wiki/Matrix_multiplication)简介）
  * 熟悉基本的 TensorFlow 数学和数组运算


```python
import tensorflow as tf
```

 ## 矢量加法

您可以对张量执行很多典型数学运算 ([TF API](https://www.tensorflow.org/api_guides/python/math_ops))。以下代码会创建和操控两个矢量（一维张量），每个矢量正好六个元素：


```python
with tf.Graph().as_default():
  # Create a six-element vector (1-D tensor).
  primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)

  # Create another six-element vector. Each element in the vector will be
  # initialized to 1. The first argument is the shape of the tensor (more
  # on shapes below).
  ones = tf.ones([6], dtype=tf.int32)

  # Add the two vectors. The resulting tensor is a six-element vector.
  just_beyond_primes = tf.add(primes, ones)

  # Create a session to run the default graph.
  with tf.Session() as sess:
    print just_beyond_primes.eval()
```

    [ 3  4  6  8 12 14]
    


```python
with tf.Graph().as_default():
  
  primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)
  
  ones = tf.ones([6], dtype=tf.int32)
  
  just_beyond_primes = tf.add(primes, ones)
  
  with tf.Session() as sess:
    
    print just_beyond_primes.eval()
```

    [ 3  4  6  8 12 14]
    

 ### 张量形状

形状用于描述张量维度的大小和数量。张量的形状表示为`列表`，其中第 `i` 个元素表示维度 `i` 的大小。列表的长度表示张量的阶（即维数）。

有关详情，请参阅 [TensorFlow 文档](https://www.tensorflow.org/programmers_guide/tensors#shape)。

以下是一些基本示例：


```python
with tf.Graph().as_default():
  # A scalar (0-D tensor).
  scalar = tf.zeros([])

  # A vector with 3 elements.
  vector = tf.zeros([3])

  # A matrix with 2 rows and 3 columns.
  matrix = tf.zeros([2, 3])

  with tf.Session() as sess:
    print 'scalar has shape', scalar.get_shape(), 'and value:\n', scalar.eval()
    print 'vector has shape', vector.get_shape(), 'and value:\n', vector.eval()
    print 'matrix has shape', matrix.get_shape(), 'and value:\n', matrix.eval()
    
```

    scalar has shape () and value:
    0.0
    vector has shape (3,) and value:
    [0. 0. 0.]
    matrix has shape (2, 3) and value:
    [[0. 0. 0.]
     [0. 0. 0.]]
    

 ### 广播

在数学中，您只能对形状相同的张量执行元素级运算（例如，*相加*和*等于*）。不过，在 TensorFlow 中，您可以对张量执行传统意义上不可行的运算。TensorFlow 支持**广播**（一种借鉴自 Numpy 的概念）。利用广播，元素级运算中的较小数组会增大到与较大数组具有相同的形状。例如，通过广播：

* 如果指令需要大小为 `[6]` 的张量，则大小为 `[1]` 或 `[]` 的张量可以作为运算数。
* 如果指令需要大小为 `[4, 6]` 的张量，则以下任何大小的张量都可以作为运算数。
  * `[1, 6]`
  * `[6]`
  * `[]`
* 如果指令需要大小为 `[3, 5, 6]` 的张量，则以下任何大小的张量都可以作为运算数。

  * `[1, 5, 6]`
  * `[3, 1, 6]`
  * `[3, 5, 1]`
  * `[1, 1, 1]`
  * `[5, 6]`
  * `[1, 6]`
  * `[6]`
  * `[1]`
  * `[]`
  
**注意：**当张量被广播时，从概念上来说，系统会**复制**其条目（出于性能考虑，实际并不复制。广播专为实现性能优化而设计）。

有关完整的广播规则集，请参阅简单易懂的 [Numpy 广播文档](http://docs.scipy.org/doc/numpy-1.10.1/user/basics.broadcasting.html)。

以下代码执行了与之前一样的张量加法，不过使用的是广播：


```python
with tf.Graph().as_default():
  # Create a six-element vector (1-D tensor).
  primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)

  # Create a constant scalar with value 1.
  ones = tf.constant(1, dtype=tf.int32)

  # Add the two tensors. The resulting tensor is a six-element vector.
  just_beyond_primes = tf.add(primes, ones)

  with tf.Session() as sess:
    print just_beyond_primes.eval()
```

    [ 3  4  6  8 12 14]
    

 ## 矩阵乘法

在线性代数中，当两个矩阵相乘时，第一个矩阵的*列*数必须等于第二个矩阵的*行*数。

- `3x4` 矩阵乘以 `4x2` 矩阵是**_有效_**的，可以得出一个 `3x2` 矩阵。
- `4x2` 矩阵乘以 `3x4` 矩阵是**_无效_**的。


```python
with tf.Graph().as_default():
  # Create a matrix (2-d tensor) with 3 rows and 4 columns.
  x = tf.constant([[5, 2, 4, 3], [5, 1, 6, -2], [-1, 3, -1, -2]],
                  dtype=tf.int32)

  # Create a matrix with 4 rows and 2 columns.
  y = tf.constant([[2, 2], [3, 5], [4, 5], [1, 6]], dtype=tf.int32)

  # Multiply `x` by `y`. 
  # The resulting matrix will have 3 rows and 2 columns.
  matrix_multiply_result = tf.matmul(x, y)

  with tf.Session() as sess:
    print matrix_multiply_result.eval()
```

    [[35 58]
     [35 33]
     [ 1 -4]]
    

 ## 张量变形

由于张量加法和矩阵乘法均对运算数施加了限制条件，TensorFlow 编程者肯定会频繁改变张量的形状。

您可以使用 `tf.reshape` 方法改变张量的形状。
例如，您可以将 8x2 张量变形为 2x8 张量或 4x4 张量：


```python
with tf.Graph().as_default():
  # Create an 8x2 matrix (2-D tensor).
  matrix = tf.constant([[1,2], [3,4], [5,6], [7,8],
                        [9,10], [11,12], [13, 14], [15,16]], dtype=tf.int32)

  # Reshape the 8x2 matrix into a 2x8 matrix.
  reshaped_2x8_matrix = tf.reshape(matrix, [2,8])
  
  # Reshape the 8x2 matrix into a 4x4 matrix
  reshaped_4x4_matrix = tf.reshape(matrix, [4,4])

  with tf.Session() as sess:
    print "Original matrix (8x2):"
    print matrix.eval()
    print "Reshaped matrix (2x8):"
    print reshaped_2x8_matrix.eval()
    print "Reshaped matrix (4x4):"
    print reshaped_4x4_matrix.eval()
```

    Original matrix (8x2):
    [[ 1  2]
     [ 3  4]
     [ 5  6]
     [ 7  8]
     [ 9 10]
     [11 12]
     [13 14]
     [15 16]]
    Reshaped matrix (2x8):
    [[ 1  2  3  4  5  6  7  8]
     [ 9 10 11 12 13 14 15 16]]
    Reshaped matrix (4x4):
    [[ 1  2  3  4]
     [ 5  6  7  8]
     [ 9 10 11 12]
     [13 14 15 16]]
    

 
此外，您还可以使用 `tf.reshape` **更改张量的维数（\'阶\'）**。
例如，您可以将 8x2 张量变形为三维 2x2x4 张量或一维 16 元素张量。


```python
with tf.Graph().as_default():
  # Create an 8x2 matrix (2-D tensor).
  matrix = tf.constant([[1,2], [3,4], [5,6], [7,8],
                        [9,10], [11,12], [13, 14], [15,16]], dtype=tf.int32)

  # Reshape the 8x2 matrix into a 3-D 2x2x4 tensor.
  reshaped_2x2x4_tensor = tf.reshape(matrix, [2,2,4])
  
  # Reshape the 8x2 matrix into a 1-D 16-element tensor.
  one_dimensional_vector = tf.reshape(matrix, [16])

  with tf.Session() as sess:
    print "Original matrix (8x2):"
    print matrix.eval()
    print "Reshaped 3-D tensor (2x2x4):"
    print reshaped_2x2x4_tensor.eval()
    print "1-D vector:"
    print one_dimensional_vector.eval()
```

    Original matrix (8x2):
    [[ 1  2]
     [ 3  4]
     [ 5  6]
     [ 7  8]
     [ 9 10]
     [11 12]
     [13 14]
     [15 16]]
    Reshaped 3-D tensor (2x2x4):
    [[[ 1  2  3  4]
      [ 5  6  7  8]]
    
     [[ 9 10 11 12]
      [13 14 15 16]]]
    1-D vector:
    [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]
    

 ### 练习 1：改变两个张量的形状，使其能够相乘。

下面两个矢量无法进行矩阵乘法运算：

  *  `a = tf.constant([5, 3, 2, 7, 1, 4])`
  *  `b = tf.constant([4, 6, 3])`

请改变这两个矢量的形状，使其成为可以进行矩阵乘法运算的运算数。
然后，对变形后的张量调用矩阵乘法运算。


```python
  # Write your code for Task 1 here.
  import tensorflow as tf 
  
  a = tf.constant([5, 3, 2, 7, 1, 4])
  
  b = tf.constant([4, 6, 3])
  
  a_reshape1 = tf.reshape(a, [2,3])
  
  b_reshape1 = tf.reshape(b, [3,1])
  
  a_reshape2 = tf.reshape(a, [6,1])
  
  b_reshape2 = tf.reshape(b, [1,3])
  
  c = tf.matmul(a_reshape1, b_reshape1)
  
  d = tf.matmul(a_reshape2, b_reshape2)
  
  with tf.Session() as sess:
      print 'c.eval():\n', c.eval()
      
      print 'd.eval():\n', d.eval()

```

    c.eval():
    [[44]
     [46]]
    d.eval():
    [[20 30 15]
     [12 18  9]
     [ 8 12  6]
     [28 42 21]
     [ 4  6  3]
     [16 24 12]]
    

 ### 解决方案

点击下方，查看解决方案。


```python
with tf.Graph().as_default(), tf.Session() as sess:
  # Task: Reshape two tensors in order to multiply them
  
  # Here are the original operands, which are incompatible
  # for matrix multiplication:
  a = tf.constant([5, 3, 2, 7, 1, 4])
  b = tf.constant([4, 6, 3])
  # We need to reshape at least one of these operands so that
  # the number of columns in the first operand equals the number
  # of rows in the second operand.

  # Reshape vector "a" into a 2-D 2x3 matrix:
  reshaped_a = tf.reshape(a, [2,3])

  # Reshape vector "b" into a 2-D 3x1 matrix:
  reshaped_b = tf.reshape(b, [3,1])

  # The number of columns in the first matrix now equals
  # the number of rows in the second matrix. Therefore, you
  # can matrix mutiply the two operands.
  c = tf.matmul(reshaped_a, reshaped_b)
  print(c.eval())

  # An alternate approach: [6,1] x [1, 3] -> [6,3]
```

    [[44]
     [46]]
    

 ## 变量、初始化和赋值

到目前为止，我们执行的所有运算都是针对静态值 (`tf.constant`) 进行的；调用 `eval()` 始终返回同一结果。在 TensorFlow 中可以定义 `Variable` 对象，它的值是可以更改的。

创建变量时，您可以明确设置一个初始值，也可以使用初始化程序（例如分布）：


```python
g = tf.Graph()
with g.as_default():
  # Create a variable with the initial value 3.
  v = tf.Variable([3])

  # Create a variable of shape [1], with a random initial value,
  # sampled from a normal distribution with mean 1 and standard deviation 0.35.
  w = tf.Variable(tf.random_normal([1], mean=1.0, stddev=0.35))
```


```python
g = tf.Graph()
with g.as_default():
  v = tf.Variable([3])
  w = tf.Variable(tf.random_normal([1], mean=1.0, stddev=0.35))
```

 TensorFlow 的一个特性是**变量初始化不是自动进行的**。例如，以下代码块会导致错误：


```python
with g.as_default():
  with tf.Session() as sess:
    try:
      v.eval()
    except tf.errors.FailedPreconditionError as e:
      print "Caught expected error: ", e
```

    Caught expected error:  Attempting to use uninitialized value Variable
    	 [[Node: _retval_Variable_0_0 = _Retval[T=DT_INT32, index=0, _device="/job:localhost/replica:0/task:0/device:CPU:0"](Variable)]]
    

 要初始化变量，最简单的方式是调用 `global_variables_initializer`。请注意 `Session.run()` 的用法（与 `eval()` 的用法大致相同）。


```python
with g.as_default():
  with tf.Session() as sess:
    initialization = tf.global_variables_initializer()
    sess.run(initialization)
    # Now, variables can be accessed normally, and have values assigned to them.
    print v.eval()
    print w.eval()

```

    [3]
    [0.43437064]
    


```python
with g.as_default():
  with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print v.eval()
    print w.eval()
```

    [3]
    [1.1014321]
    

 初始化后，变量的值保留在同一会话中（不过，当您启动新会话时，需要重新初始化它们）：


```python
with g.as_default():
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # These three prints will print the same value.
    print w.eval()
    print w.eval()
    print w.eval()
```

    [0.78335255]
    [0.78335255]
    [0.78335255]
    

 要更改变量的值，请使用 `assign` 指令。请注意，仅创建 `assign` 指令不会起到任何作用。和初始化一样，您必须`运行`赋值指令才能更新变量值：


```python
with g.as_default():
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # This should print the variable's initial value.
    print v.eval()

    assignment = tf.assign(v, [7])
    # The variable has not been changed yet!
    print v.eval()

    # Execute the assignment op.
    sess.run(assignment)
    # Now the variable is updated.
    print v.eval()
```

    [3]
    [3]
    [7]
    


```python

with g.as_default():
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print v.eval()
    
    assign = tf.assign(v, [888])
    sess.run(assign)
    print v.eval()
    
```

    [3]
    [888]
    

 还有很多关于变量的内容我们并未在这里提及，例如加载和存储。要了解详情，请参阅 [TensorFlow 文档](https://www.tensorflow.org/programmers_guide/variables)。

 ### 练习 2：模拟投掷两个骰子 10 次。

创建一个骰子模拟，在模拟中生成一个 `10x3` 二维张量，其中：

  * 列 `1` 和 `2` 均存储一个骰子的一次投掷值。
  * 列 `3` 存储同一行中列 `1` 和 `2` 的值的总和。

例如，第一行中可能会包含以下值：

  * 列 `1` 存储 `4`
  * 列 `2` 存储 `3`
  * 列 `3` 存储 `7`

要完成此任务，您需要浏览 [TensorFlow 文档](https://www.tensorflow.org/api_guides/python/array_ops)。


```python
# Write your code for Task 2 here.

import tensorflow as tf 

with tf.Graph().as_default():
  with tf.Session() as sess:
    
    d1 = tf.Variable(tf.random_uniform([10, 1],minval=1, maxval=7,dtype=tf.int32))
    d2 = tf.Variable(tf.random_uniform([10, 1], minval=1, maxval=7,dtype=tf.int32))
    d3 = tf.add(d1, d2)
             
    result = tf.concat(values=[d1, d2, d3], axis=1)      
    sess.run(tf.global_variables_initializer())
    print(result.eval())
```

    [[ 3  5  8]
     [ 3  6  9]
     [ 6  5 11]
     [ 4  2  6]
     [ 3  5  8]
     [ 4  2  6]
     [ 4  6 10]
     [ 1  6  7]
     [ 2  3  5]
     [ 2  4  6]]
    

 ### 解决方案

点击下方，查看解决方案。


```python
with tf.Graph().as_default(), tf.Session() as sess:
  # Task 2: Simulate 10 throws of two dice. Store the results
  # in a 10x3 matrix.

  # We're going to place dice throws inside two separate
  # 10x1 matrices. We could have placed dice throws inside
  # a single 10x2 matrix, but adding different columns of
  # the same matrix is tricky. We also could have placed
  # dice throws inside two 1-D tensors (vectors); doing so
  # would require transposing the result.
  dice1 = tf.Variable(tf.random_uniform([10, 1],
                                        minval=1, maxval=7,
                                        dtype=tf.int32))
  dice2 = tf.Variable(tf.random_uniform([10, 1],
                                        minval=1, maxval=7,
                                        dtype=tf.int32))

  # We may add dice1 and dice2 since they share the same shape
  # and size.
  dice_sum = tf.add(dice1, dice2)

  # We've got three separate 10x1 matrices. To produce a single
  # 10x3 matrix, we'll concatenate them along dimension 1.
  resulting_matrix = tf.concat(
      values=[dice1, dice2, dice_sum], axis=1)

  # The variables haven't been initialized within the graph yet,
  # so let's remedy that.
  sess.run(tf.global_variables_initializer())

  print(resulting_matrix.eval())
```

    [[ 6  5 11]
     [ 6  2  8]
     [ 5  6 11]
     [ 2  3  5]
     [ 4  3  7]
     [ 3  4  7]
     [ 1  1  2]
     [ 5  4  9]
     [ 5  4  9]
     [ 5  5 10]]
    
