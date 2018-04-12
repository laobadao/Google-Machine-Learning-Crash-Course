
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

 # 合成特征和离群值

 **学习目标：**
  * 创建一个合成特征，即另外两个特征的比例
  * 将此新特征用作线性回归模型的输入
  * 通过识别和截取（移除）输入数据中的离群值来提高模型的有效性

 我们来回顾下之前的“使用 TensorFlow 的基本步骤”练习中的模型。

首先，我们将加利福尼亚州住房数据导入 *Pandas* `DataFrame` 中：

 ## 设置


```python
import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10 # 显示时更紧凑 只显示 10 行
pd.options.display.float_format = '{:.1f}'.format # 对显示的数据格式化，只显示 1 位小数

california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/ml_universities/california_housing_train.csv", sep=',')

# 重新索引 pandas 对象的一个重要方法是 reindex，其作用是创建一个新对象，它的数据符合新的索引。
california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))

california_housing_dataframe["median_house_value"] /= 1000.0 # median_house_value 这列的数据都除以 1000.0

california_housing_dataframe
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16944</th>
      <td>-124.1</td>
      <td>40.7</td>
      <td>23.0</td>
      <td>580.0</td>
      <td>117.0</td>
      <td>320.0</td>
      <td>109.0</td>
      <td>4.2</td>
      <td>130.6</td>
    </tr>
    <tr>
      <th>7216</th>
      <td>-118.3</td>
      <td>33.9</td>
      <td>31.0</td>
      <td>3757.0</td>
      <td>1102.0</td>
      <td>3288.0</td>
      <td>964.0</td>
      <td>1.9</td>
      <td>137.5</td>
    </tr>
    <tr>
      <th>14798</th>
      <td>-122.2</td>
      <td>40.2</td>
      <td>30.0</td>
      <td>744.0</td>
      <td>156.0</td>
      <td>410.0</td>
      <td>165.0</td>
      <td>2.2</td>
      <td>63.2</td>
    </tr>
    <tr>
      <th>5197</th>
      <td>-118.1</td>
      <td>33.8</td>
      <td>45.0</td>
      <td>3087.0</td>
      <td>574.0</td>
      <td>1474.0</td>
      <td>567.0</td>
      <td>5.5</td>
      <td>227.6</td>
    </tr>
    <tr>
      <th>13799</th>
      <td>-122.0</td>
      <td>37.4</td>
      <td>16.0</td>
      <td>3716.0</td>
      <td>916.0</td>
      <td>1551.0</td>
      <td>759.0</td>
      <td>4.5</td>
      <td>323.6</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>540</th>
      <td>-117.0</td>
      <td>33.0</td>
      <td>13.0</td>
      <td>4595.0</td>
      <td>567.0</td>
      <td>1643.0</td>
      <td>544.0</td>
      <td>7.8</td>
      <td>362.3</td>
    </tr>
    <tr>
      <th>16293</th>
      <td>-122.5</td>
      <td>37.7</td>
      <td>29.0</td>
      <td>3795.0</td>
      <td>675.0</td>
      <td>2494.0</td>
      <td>696.0</td>
      <td>5.3</td>
      <td>260.3</td>
    </tr>
    <tr>
      <th>11673</th>
      <td>-121.3</td>
      <td>38.7</td>
      <td>23.0</td>
      <td>2145.0</td>
      <td>340.0</td>
      <td>1022.0</td>
      <td>349.0</td>
      <td>4.2</td>
      <td>125.4</td>
    </tr>
    <tr>
      <th>7491</th>
      <td>-118.4</td>
      <td>34.1</td>
      <td>52.0</td>
      <td>1902.0</td>
      <td>488.0</td>
      <td>848.0</td>
      <td>478.0</td>
      <td>3.0</td>
      <td>175.0</td>
    </tr>
    <tr>
      <th>6438</th>
      <td>-118.3</td>
      <td>34.2</td>
      <td>30.0</td>
      <td>2180.0</td>
      <td>369.0</td>
      <td>1050.0</td>
      <td>390.0</td>
      <td>6.4</td>
      <td>277.6</td>
    </tr>
  </tbody>
</table>
<p>17000 rows × 9 columns</p>
</div>



 接下来，我们将设置输入函数，并针对模型训练来定义该函数：
 
 [Python for Data Analysis v2 | Notes_ Chapter_5 pandas 入门](https://blog.csdn.net/junjun_zhao/article/details/79861041)


```python
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.
        训练一个特征的线性回归模型。
        
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    
    # Convert pandas data into a dict of np arrays. 将 pandas 数据转换为 np 数组的字典。也就是 从字典重 将所有 数据 value 取出来 然后转化为 数组                                           
    features = {key:np.array(value) for key, value in dict(features).items()}
    
    # Construct a dataset, and configure batching/repeating  设置 batch_size 和 迭代次数 num_epochs
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified 随机打乱数据
    if shuffle:
      ds = ds.shuffle(buffer_size=10000)
    
    # Return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels
```


```python
def train_model(learning_rate, steps, batch_size, input_feature):
  """Trains a linear regression model. 训练线性回归模型。
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
      一个非零的`int`，即训练步骤的总数。 训练步骤包括使用单个批次的前向和后向传播。
      
    batch_size: A non-zero `int`, the batch size.
    input_feature: A `string` specifying a column from `california_housing_dataframe`
      to use as input feature.
      
  Returns:
    A Pandas `DataFrame` containing targets and the corresponding predictions done
    after training the model.
    包含在训练完模型后的目标和相应预测的 pandas `DataFrame`。
    
  """
  
  periods = 10
  steps_per_period = steps / periods

  my_feature = input_feature
  # 取出特征数据   
  my_feature_data = california_housing_dataframe[[my_feature]].astype('float32')
  # 取出标签数据
  my_label = "median_house_value"
  targets = california_housing_dataframe[my_label].astype('float32')

  # Create input functions
  training_input_fn = lambda: my_input_fn(my_feature_data, targets, batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)
  
  # Create feature columns
  feature_columns = [tf.feature_column.numeric_column(my_feature)]
    
  # Create a linear regressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=feature_columns,
      optimizer=my_optimizer
  )

  # Set up to plot the state of our model's line each period.
  plt.figure(figsize=(15, 6))
  plt.subplot(1, 2, 1)
  plt.title("Learned Line by Period")
  plt.ylabel(my_label)
  plt.xlabel(my_feature)
  sample = california_housing_dataframe.sample(n=300)
  plt.scatter(sample[my_feature], sample[my_label])
  colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...") 
  print("RMSE 均方根误差 (on training data):") 
  root_mean_squared_errors = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period,
    )
    # Take a break and compute predictions.
    predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
    predictions = np.array([item['predictions'][0] for item in predictions])
    
    # Compute loss.
    root_mean_squared_error = math.sqrt(
      metrics.mean_squared_error(predictions, targets))
    # Occasionally print the current loss.
    print(" period %02d : %0.2f" % (period, root_mean_squared_error)) 
    # Add the loss metrics from this period to our list.
    root_mean_squared_errors.append(root_mean_squared_error)
    # Finally, track the weights and biases over time.
    # Apply some math to ensure that the data and line are plotted neatly.
    y_extents = np.array([0, sample[my_label].max()])
    
    weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
    bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')
    
    x_extents = (y_extents - bias) / weight
    x_extents = np.maximum(np.minimum(x_extents,
                                      sample[my_feature].max()),
                           sample[my_feature].min())
    y_extents = weight * x_extents + bias
    plt.plot(x_extents, y_extents, color=colors[period]) 
  print("Model training finished.") 

  # Output a graph of loss metrics over periods.
  plt.subplot(1, 2, 2)
  plt.ylabel('RMSE')
  plt.xlabel('Periods')
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(root_mean_squared_errors)

  # Create a table with calibration data.
  calibration_data = pd.DataFrame()
  calibration_data["predictions"] = pd.Series(predictions)
  calibration_data["targets"] = pd.Series(targets)
  display.display(calibration_data.describe())

  print("Final RMSE (on training data): %0.2f" % root_mean_squared_error) 
  
  return calibration_data
```

 ## 任务 1：尝试合成特征

`total_rooms` 和 `population` 特征都会统计指定街区的相关总计数据。

但是，如果一个街区比另一个街区的人口更密集，会怎么样？我们可以创建一个合成特征（即 `total_rooms` 与 `population` 的比例）来探索街区人口密度与房屋价值中位数之间的关系。

在以下单元格中，创建一个名为 `rooms_per_person` 的特征，并将其用作 `train_model()` 的 `input_feature`。

通过调整学习速率，您使用这一特征可以获得的最佳效果是什么？（效果越好，回归线与数据的拟合度就越高，最终 RMSE 也会越低。） 均方根误差 (RMSE, Root Mean Squared Error)

 **注意**：在下面添加一些代码单元格可能有帮助，这样您就可以尝试几种不同的学习速率并比较结果。要添加新的代码单元格，请将光标悬停在该单元格中心的正下方，然后点击**代码**。


```python
#
# YOUR CODE HERE
#
california_housing_dataframe["rooms_per_person"] = california_housing_dataframe["total_rooms"] / california_housing_dataframe["population"]

calibration_data = train_model(
    learning_rate=0.05,
    steps=500,
    batch_size=5,
    input_feature="rooms_per_person"
)
```

    Training model...
    RMSE (on training data):
     period 00 : 212.77
     period 01 : 189.68
     period 02 : 169.67
     period 03 : 154.33
     period 04 : 141.13
     period 05 : 135.27
     period 06 : 131.62
     period 07 : 130.43
     period 08 : 130.53
     period 09 : 131.73
    Model training finished.
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>predictions</th>
      <th>targets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>17000.0</td>
      <td>17000.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>199.0</td>
      <td>207.3</td>
    </tr>
    <tr>
      <th>std</th>
      <td>90.4</td>
      <td>116.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>47.3</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>163.6</td>
      <td>119.4</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>196.0</td>
      <td>180.4</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>223.5</td>
      <td>265.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4315.6</td>
      <td>500.0</td>
    </tr>
  </tbody>
</table>
</div>


    Final RMSE (on training data): 131.73
    


![png](output_12_3.png)


 ### 解决方案

点击下方即可查看解决方案。


```python
california_housing_dataframe["rooms_per_person"] = (
    california_housing_dataframe["total_rooms"] / california_housing_dataframe["population"])

calibration_data = train_model(
    learning_rate=0.05,
    steps=500,
    batch_size=5,
    input_feature="rooms_per_person")
```

 ## 任务 2：识别离群值

我们可以通过创建预测值与目标值的散点图来可视化模型效果。理想情况下，这些值将位于一条完全相关的对角线上。

使用您在任务 1 中训练过的人均房间数模型，并使用 Pyplot 的 `scatter()` 创建预测值与目标值的散点图。

您是否看到任何异常情况？通过查看 `rooms_per_person` 中值的分布情况，将这些异常情况追溯到源数据。


```python
# YOUR CODE HERE
plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.scatter(calibration_data["predictions"], calibration_data['targets'])
```




    <matplotlib.collections.PathCollection at 0x7f2eba697c50>




![png](output_16_1.png)


 ### 解决方案

点击下方即可查看解决方案。


```python
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.scatter(calibration_data["predictions"], calibration_data["targets"])
```

 校准数据显示，大多数散点与一条线对齐。这条线几乎是垂直的，我们稍后再讲解。现在，我们重点关注偏离这条线的点。我们注意到这些点的数量相对较少。

如果我们绘制 `rooms_per_person` 的直方图，则会发现我们的输入数据中有少量离群值：


```python
plt.subplot(1, 2, 2)
_ = california_housing_dataframe["rooms_per_person"].hist()
```


![png](output_20_0.png)


 ## 任务 3：截取离群值

看看您能否通过将 `rooms_per_person` 的离群值设置为相对合理的最小值或最大值来进一步改进模型拟合情况。

以下是一个如何将函数应用于 Pandas `Series` 的简单示例，供您参考：

    clipped_feature = my_dataframe["my_feature_name"].apply(lambda x: max(x, 0))

上述 `clipped_feature` 没有小于 `0` 的值。


```python
# YOUR CODE HERE
```

 ### 解决方案

点击下方即可查看解决方案。

 我们在任务 2 中创建的直方图显示，大多数值都小于 `5`。我们将 `rooms_per_person` 的值截取为 5，然后绘制直方图以再次检查结果。


```python
california_housing_dataframe["rooms_per_person"] = (
    california_housing_dataframe["rooms_per_person"]).apply(lambda x: min(x, 5))

_ = california_housing_dataframe["rooms_per_person"].hist()
```


![png](output_25_0.png)


 为了验证截取是否有效，我们再训练一次模型，并再次输出校准数据：


```python
calibration_data = train_model(
    learning_rate=0.05,
    steps=500,
    batch_size=5,
    input_feature="rooms_per_person")
```

    Training model...
    RMSE (on training data):
     period 00 : 212.84
     period 01 : 189.08
     period 02 : 166.83
     period 03 : 147.59
     period 04 : 132.60
     period 05 : 120.94
     period 06 : 113.69
     period 07 : 110.09
     period 08 : 108.71
     period 09 : 108.26
    Model training finished.
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>predictions</th>
      <th>targets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>17000.0</td>
      <td>17000.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>194.9</td>
      <td>207.3</td>
    </tr>
    <tr>
      <th>std</th>
      <td>50.6</td>
      <td>116.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>46.5</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>162.6</td>
      <td>119.4</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>194.9</td>
      <td>180.4</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>222.4</td>
      <td>265.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>431.0</td>
      <td>500.0</td>
    </tr>
  </tbody>
</table>
</div>


    Final RMSE (on training data): 108.26
    


![png](output_27_3.png)



```python
_ = plt.scatter(calibration_data["predictions"], calibration_data["targets"])
```


![png](output_28_0.png)

