---
layout: post
title: Building Simple Neural Network with Keras
---
***

The problem: MNIST handwritten digit classification
MNIST data-set is classic deep learning problem. It's a collection of handwritten digits from 0 to 9.

![png](/images/Keras_image/iu.png)

Keras is simple and powerfull deep learning library for Python. You can learn more by reading the <a href='https://keras.io/getting_started/intro_to_keras_for_engineers/'>documentation</a>.

![png](/images/Keras_image/iu-2.png)

Let's start!


```python
import numpy as np
import keras
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
```

## Data set

Uploading the data set. You can download it from here: http://pjreddie.com/projects/mnist-in-csv/


```python
# let's upload train data
train_data_file = open('data/mnist/mnist_train.csv','r')
train_data_list = train_data_file.readlines()
train_data_file.close()

# # let's upload test data
test_data_file = open('data/mnist/mnist_test.csv','r')
test_data_list = test_data_file.readlines()
test_data_file.close()
```


```python
print('Number of training examples: ',len(train_data_list))
print('Number of test examples: ',len(test_data_list))
```

    Number of training examples:  60000
    Number of test examples:  10000


## Data Preparation

Let's split labels and features into separate data sets.


```python
# y - targets
# X - features
y_train = []
X_train = []

for record in range(len(train_data_list)):
    y_train.append(train_data_list[record][0])
    values = train_data_list[record].split(',')
    X_train.append(values[1:])

y_test = []
X_test = []

for record in range(len(test_data_list)):
    y_test.append(test_data_list[record][0])
    values = test_data_list[record].split(',')
    X_test.append(values[1:])
```


```python
# converting to numpy array
y_train = np.asfarray(y_train)
X_train = np.asfarray(X_train)

y_test = np.asfarray(y_test)
X_test = np.asfarray(X_test)
```


```python
train_images = X_train.reshape((-1, 784))
test_images = X_test.reshape((-1, 784))

# check the shapes
print('y_train shape:',y_train.shape)
print('X_train shape: ',X_train.shape)

print('y_test shape:',y_test.shape)
print('X_test shape: ',X_test.shape)
```

    y_train shape: (60000,)
    X_train shape:  (60000, 784)
    y_test shape: (10000,)
    X_test shape:  (10000, 784)


Then we normalize our data. Instead of having pixel values from [0-255] we center them from [-0.5 to 0.5]. Usually smaller and centered values are better to train.


```python
# Normalize the images.
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5
```

## Building the Model

Keras allows to build **Sequential** and **Functional** models. Sequential model is the simplest model where layers of neurons stacked and fuly connected. Functional model is more customizable. Here we're going to build Sequential model.


```python
# instantiate model
from keras.models import Sequential
from keras.layers import Dense

model = Sequential([
    Dense(64,activation='relu'),
    Dense(64,activation='relu'),
    Dense(10,activation='softmax')
])
```

First and second layers, each have 64 nodes with <a href='https://en.wikipedia.org/wiki/Rectifier_(neural_networks)'>ReLU</a> activation function. Output layer has 10 nodes, one for each label with a <a href='https://en.wikipedia.org/wiki/Softmax_function'>Softmax</a> activation function.

## Compile the Model

Now we need to compile our model before we start training. We need to define 3 main key factors:
* Optimizer - gradient descent
* Loss function
* Metric

Keras has many <a href='https://keras.io/api/optimizers/'>optimizers</a>. In our model we will use <a href='https://arxiv.org/abs/1412.6980'>**Adam** - gradient based optimization</a>. 
For the Loss function **Cross-Entropy Loss**. To learn more about loss functions, go to Keras documentation: <a href='https://keras.io/api/losses/'>Keras' loss functions</a>. As for the metric we'll use **accuracy**.



```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

## Training the Model


```python
from keras.utils import to_categorical

model.fit(
    x=train_images, #train data-set
    y=to_categorical(y_train), #labels
    epochs=5,
    batch_size=32
)
```

    Epoch 1/5
    60000/60000 [==============================] - 4s 72us/step - loss: 0.0946 - accuracy: 0.9700
    Epoch 2/5
    60000/60000 [==============================] - 4s 69us/step - loss: 0.0827 - accuracy: 0.9734
    Epoch 3/5
    60000/60000 [==============================] - 4s 69us/step - loss: 0.0774 - accuracy: 0.9752
    Epoch 4/5
    60000/60000 [==============================] - 4s 68us/step - loss: 0.0691 - accuracy: 0.9778
    Epoch 5/5
    60000/60000 [==============================] - 4s 69us/step - loss: 0.0645 - accuracy: 0.9790





    <keras.callbacks.callbacks.History at 0x7f7ebce536d0>



Great! After 5 epochs of training we achieved 0.9790 accuracy. It may look promising but it doesn't tell us much. We need to test the model.

## Testing the Model


```python
model.evaluate(
  test_images,
  to_categorical(y_test)
)
```

    10000/10000 [==============================] - 0s 30us/step





    [0.088274097120855, 0.9717000126838684]



After testing, our model's loss is 0.088 and accuracy is 0.9717. Not bad at all, slightly lower accuracy than on training data.

## Experiment with Model

Let's try out different parameters to compare the results.

### Number of epochs?


```python
model = Sequential([
    Dense(64,activation='relu'),
    Dense(64,activation='relu'),
    Dense(10,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


from keras.utils import to_categorical

model.fit(
    x=train_images, #train data-set
    y=to_categorical(y_train), #labels
    epochs=10,
    batch_size=32
)

print('test accuracy: ')

model.evaluate(
  test_images,
  to_categorical(y_test)
)

```

    Epoch 1/10
    60000/60000 [==============================] - 4s 71us/step - loss: 0.3518 - accuracy: 0.8953
    Epoch 2/10
    60000/60000 [==============================] - 4s 72us/step - loss: 0.1812 - accuracy: 0.9449
    Epoch 3/10
    60000/60000 [==============================] - 4s 70us/step - loss: 0.1415 - accuracy: 0.9566
    Epoch 4/10
    60000/60000 [==============================] - 4s 71us/step - loss: 0.1192 - accuracy: 0.9628
    Epoch 5/10
    60000/60000 [==============================] - 4s 71us/step - loss: 0.1042 - accuracy: 0.9669
    Epoch 6/10
    60000/60000 [==============================] - 4s 71us/step - loss: 0.0952 - accuracy: 0.9698
    Epoch 7/10
    60000/60000 [==============================] - 5s 75us/step - loss: 0.0848 - accuracy: 0.9731
    Epoch 8/10
    60000/60000 [==============================] - 5s 79us/step - loss: 0.0773 - accuracy: 0.9751
    Epoch 9/10
    60000/60000 [==============================] - 5s 77us/step - loss: 0.0714 - accuracy: 0.9772
    Epoch 10/10
    60000/60000 [==============================] - 5s 77us/step - loss: 0.0675 - accuracy: 0.9774
    test accuracy: 
    10000/10000 [==============================] - 0s 39us/step





    [0.10760164717165753, 0.9682999849319458]



Looks like accuracy of the model slightly deteriorated with more iteration. May be **overfitting**? 

### Network Depth?


```python
# more layers
model = Sequential([
    Dense(64,activation='relu'),
    Dense(64,activation='relu'),
    Dense(64,activation='relu'),
    Dense(64,activation='relu'),
    Dense(10,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


from keras.utils import to_categorical

model.fit(
    x=train_images, #train data-set
    y=to_categorical(y_train), #labels
    epochs=5,
    batch_size=32
)


model.evaluate(
  test_images,
  to_categorical(y_test)
)

```

    Epoch 1/5
    60000/60000 [==============================] - 5s 83us/step - loss: 0.3504 - accuracy: 0.8903
    Epoch 2/5
    60000/60000 [==============================] - 5s 82us/step - loss: 0.1767 - accuracy: 0.9449
    Epoch 3/5
    60000/60000 [==============================] - 5s 87us/step - loss: 0.1434 - accuracy: 0.9550
    Epoch 4/5
    60000/60000 [==============================] - 5s 86us/step - loss: 0.1182 - accuracy: 0.9631
    Epoch 5/5
    60000/60000 [==============================] - 5s 79us/step - loss: 0.1049 - accuracy: 0.9675
    10000/10000 [==============================] - 0s 34us/step





    [0.12213691611355171, 0.9623000025749207]



### Different Activation: Sigmoid?


```python

model = Sequential([
    Dense(64,activation='sigmoid'),
    Dense(64,activation='sigmoid'),
    Dense(10,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


from keras.utils import to_categorical

model.fit(
    x=train_images, #train data-set
    y=to_categorical(y_train), #labels
    epochs=5,
    batch_size=32
)

print('test accuracy: ')

model.evaluate(
  test_images,
  to_categorical(y_test)
)
```

    Epoch 1/5
    60000/60000 [==============================] - 5s 79us/step - loss: 0.5703 - accuracy: 0.8563
    Epoch 2/5
    60000/60000 [==============================] - 4s 73us/step - loss: 0.2328 - accuracy: 0.9316
    Epoch 3/5
    60000/60000 [==============================] - 4s 74us/step - loss: 0.1740 - accuracy: 0.9487
    Epoch 4/5
    60000/60000 [==============================] - 4s 69us/step - loss: 0.1411 - accuracy: 0.9581
    Epoch 5/5
    60000/60000 [==============================] - 4s 72us/step - loss: 0.1184 - accuracy: 0.9652
    test accuracy: 
    10000/10000 [==============================] - 0s 35us/step





    [0.13002270174250008, 0.9621000289916992]



## Conclusion

You can tune your parameters and hyper-parameters of your model to achieve desired outcome. We have implemented 4 layer (input, 2 hidden and output) neural network using <a href='https://keras.io'>Keras</a>, and achived 97% accuracy on train data-set, 97% on test data-set as well.

As you can see above, we can play with a model with different parameters and see the results. At each setting, results vary. We should always test our model, and try different parameters. 