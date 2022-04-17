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


### Import TensorFlow into your program:
~~~python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
~~~

### Load and prepare the MNIST dataset.
~~~python
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
~~~
