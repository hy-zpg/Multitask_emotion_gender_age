from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Activation, add, Dense, Flatten, Dropout, Multiply, Embedding, Lambda
from keras.layers import Conv2D, MaxPooling2D,PReLU
from keras import backend as K
import numpy as np
import sys
from keras.callbacks import *
import TYY_callbacks
from keras.optimizers import SGD, Adam


batch_size = 128
num_classes = 50
epochs = 10
isCenterloss = True
#isCenterloss = False



# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print ('y_train shape.',y_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Maintain single value ground truth labels for center loss inputs
# Because Embedding layer only accept index as inputs instead of one-hot vector
y_train_value = y_train
y_test_value = y_test

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print (y_train.shape)

def base_model():
  inputs = Input(shape=(28,28,1))
  x = Conv2D(32, (3,3))(inputs)
  x = PReLU()(x)
  x = Conv2D(32, (3,3))(x)
  x = PReLU()(x)
  x = Conv2D(64, (3,3))(x)
  x = PReLU()(x)
  x = Conv2D(64, (5,5))(x)
  x = PReLU()(x)
  x = Conv2D(128, (5,5))(x)
  x = PReLU()(x)
  x = Conv2D(128, (5,5))(x)
  x = PReLU()(x)
  x = Flatten()(x)
  x = Dense(2)(x)
  ip1 = PReLU(name='ip1')(x)
  ip2 = Dense(num_classes, activation='softmax')(ip1)
  model = Model(inputs=inputs, outputs=[ip1,ip2])
  return model


'''model = Model(inputs=inputs, outputs=[ip2])
model.compile(loss="categorical_crossentropy",
              optimizer=SGD(lr=0.05),
              metrics=['accuracy'])'''
inputs = Input(shape=(28,28,1))
base_model = base_model()
[ip1,ip2] = base_model(inputs)

if isCenterloss:
  lambda_c = 0.2
  input_target = Input(shape=(1,)) # single value ground truth labels as inputs
  centers = Embedding(10,2)(input_target)
  l2_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:,0]),1,keepdims=True),name='l2_loss')([ip1,centers])
  model_centerloss = Model(inputs=[inputs,input_target],outputs=[ip2,l2_loss])        
  model_centerloss.compile(optimizer=SGD(lr=0.05), loss=["categorical_crossentropy", lambda y_true,y_pred: y_pred],loss_weights=[1,lambda_c],metrics=['accuracy'])


# prepare callback
histories = TYY_callbacks.Histories(isCenterloss)
 
# fit
if isCenterloss:
  random_y_train = np.random.rand(x_train.shape[0],1)
  random_y_test = np.random.rand(x_test.shape[0],1)
  print ('input shape:',x_train.shape,y_train_value.shape,'output shape:',y_train.shape,random_y_train.shape)
  model_centerloss.fit([x_train,y_train_value], [y_train, random_y_train], batch_size=batch_size, epochs=epochs, verbose=1, validation_data=([x_test,y_test_value], [y_test,random_y_test]), callbacks=[histories])

else:
  model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test,y_test), callbacks=[histories])

