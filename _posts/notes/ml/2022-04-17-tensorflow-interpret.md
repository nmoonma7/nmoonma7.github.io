---
layout: post
title: TensorFlow - TensorFlow 2 quickstart for experts
description: >
  TensorFlow 공식 가이드에 대한 해석. TensorFlow 2 quickstart for experts  

sitemap: false
hide_last_modified: true
categories:
    - notes
    - ml
---

# tensorflow

* toc
{:toc}


### Import TensorFlow into your program
~~~python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
~~~

### Load and prepare the MNIST dataset
~~~python
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

# x_train.sahpe 
# (60000,28,28) -> (60000,28,28,1)
~~~

tf.newaxis는 한차원 추가하는 역할. 예를 들어  
[[1,2],[3,4]] -> [[[1],[2]],[[3],[4]]]  
shape는 (2,2) -> (2,2,1)  

차원 보는법. 예를 들어  
[[[1],[2]],[[3],[4]],[[5],[6]]]  -> (3,2,1)  
가장 바깥 쪽 괄호 안에 있는 원소 갯수 3개임 : 3  
괄호 벗겨내면  
[[1],[2]]  
그다음 가장 바깥쪽 괄호 안에 있는 원소 갯수 2개임 :2  
괄호 벗겨내면  
[1]  
그다음 가장 바깥 쪽 괄호 안에 있는 원소 갯수 1개임 :1  

### Use tf.data to batch and shuffle the dataset

~~~python
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
~~~

shuffle(buffer_size, seed=None, reshuffle_each_iteration=None) - dataset을 섞는 method

인수인 buffer_size만큼 가져와서 셔플을 하기 때문에 이상적으로는 전체 dataset의 크기보다 큰 buffer-size가 필요하다.   
100개의 데이터 중 10개가 타겟이라고 하였을 때, buffer_size를 10으로 할 경우, 10개의 타겟이 몰려 있었다면 타겟끼리 자리 바꿈만 하게 됨. 

### Build the tf.keras model using the Keras model subclassing API

~~~python
class MyModel(Model):
  def __init__(self):
    #super(MyModel, self).__init__()
    super().__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10)

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

# Create an instance of the model
model = MyModel()
~~~