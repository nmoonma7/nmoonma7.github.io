---
layout: post
title: (TensorFlow) TensorFlow 2 quickstart for experts
description: >
  TensorFlow 공식 가이드(TensorFlow 2 quickstart for experts)에 대한 해석.   

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

인수인 buffer_size만큼 가져와서 셔플을 하기 때문에 이상적으로는 전체 dataset의 크기보다 큰 buffer-size가 필요하다. 100개의 데이터 중 10개가 타겟이라고 하였을 때, buffer_size를 10으로 할 경우, 10개의 타겟이 몰려 있었다면 타겟끼리 자리 바꿈만 하게 됨. 

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

여기서 self는  
- The 'self' parameter is a reference to the current instance of the class, 
and is used to access variables that belongs to the class.

부모클래스.__init__(self) 과 super().__init__()은 같은 역할

- super(MyModel, self).__init__() 는 python 2,3 에서 작동  
- super().__init__() 는 python 3 에서 작동

self.conv1 = Conv2D(32, 3, activation='relu')
32개는 filter 수.
위에 같은 경우는, 3x3x1 필터 32개가 돌아다닌다. 
만약 RGB 3채널이였을 경우 3X3X3 형태의 필터였을 것이다

### Choose an optimizer and loss function for training
~~~python
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

~~~

categorical cross entropy vs SparseCategoricalCrossentropy  
- If your $Y_{i}$ 's are one-hot encoded, use categorical_crossentropy. 
Examples (for a 3-class classification): [1,0,0] , [0,1,0], [0,0,1]  
- But if your Yi's are integers, use sparse_categorical_crossentropy. 
Examples for above 3-class classification problem: [1] , [2], [3]

Sparse Categorical Crossentropy의 sparse는 어떤 의미일까?
이름이 왜 그런지는 정확히 모른다. 단순히 integer label을 sparse(one_hot)형태로 바꿔서 계산해야된다는 의미 일까?

from_logits 는 뭘까?
- from_logits=True means the input to crossEntropy layer is normal tensor/logits,   
- from_logits=False, means the input is a probability and usually you should have some softmax activation in your last layer.

통계적 의미의 logit과 딥러닝에서 사용되는 logit의 의미는 다른다고 한다. [from_logit에 관한 설명](https://stackoverflow.com/questions/41455101/what-is-the-meaning-of-the-word-logits-in-tensorflow/52111173#52111173)

cross entropy는 뭔가
- sigma P(x) log(Q(x))
P(x) 는 현재 데이터 분포 Q(x)는 예측 분포
one hot encoded 된 데이터로 예시를 들면
S(y)= yhat = [0.7, 0.2, 0.1] 
  L = True = [1.0 ,0,0, 0.0]
D(S, L) = - sigma Li*log(Si) 형태가 됌

optimizer에 대한 정리 필요 
BGD 같은 경우는 전체 데이터 gradient 평균을 빼는 형식으로 업데이트, 
SGD는 한 점의 gradient를 빼는 형식으로 업데이트 
https://towardsdatascience.com/stochastic-batch-and-mini-batch-gradient-descent-demystified-8b28978f7f5

adam 에 대한 개념 
https://onevision.tistory.com/entry/Optimizer-%EC%9D%98-%EC%A2%85%EB%A5%98%EC%99%80-%ED%8A%B9%EC%84%B1-Momentum-RMSProp-Adam
