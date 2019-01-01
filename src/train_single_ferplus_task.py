import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import keras.backend as K
from keras_squeezenet import SqueezeNet
tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth=True
sess = tf.Session(config=tfconfig)
K.set_session(sess)
import numpy as np
import glob
import pandas as pd
import random
from PIL import Image
import sys
import time
from  models_center_loss import VGG16, Xception
import generate_data_ferplus as gd_emotion
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.generic_utils import Progbar
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint, CSVLogger
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, Dense, Flatten
from keras.layers import Reshape, TimeDistributed, Activation
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import Adam, RMSprop
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, Dropout, GlobalAveragePooling2D, merge
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense, Flatten
from keras.regularizers import l2
import keras.losses as losses
from keras.layers import  add, Multiply, Embedding, Lambda
from keras.layers import MaxPooling2D,PReLU
from keras.utils.vis_utils import plot_model 
from tqdm import tqdm
import pickle
from keras.optimizers import SGD, Adam

isCenterloss = True
crop_size=224
batch_size = 32
num_outputs_emotion=8


image_batch = Input(shape=(crop_size, crop_size, 1), name='in_t1')
my_model =Xception((crop_size, crop_size, 1),num_outputs_emotion)
emotion_feature,emotion_out = my_model(image_batch)


img_file = '/home/yanhong/Downloads/next_step/Xception/datasets/ferplus/ferplus_images/'
label_file = '/home/yanhong/Downloads/next_step/Xception/datasets/ferplus/labels_list/'
train_list_data,validation_list_data = gd_emotion.split_train_validation(label_file,split=0.8)
gen_train_emotion = gd_emotion.generate_data(img_file,train_list_data, batch_size, crop_size)
gen_valid_emotion = gd_emotion.generate_data(img_file,validation_list_data, batch_size, crop_size)


if  isCenterloss:
    lambda_c = 0.2
    input_target = Input(shape=(1,)) # single value ground truth labels as inputs
    centers = Embedding(num_outputs_emotion,2048)(input_target)# 1028 is the feature size
    l2_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:,0]),1,keepdims=True),name='l2_loss')([emotion_feature,centers])
    model_centerloss = Model(inputs=[image_batch,input_target],outputs=[emotion_out,l2_loss])   
    model_centerloss.summary()     
    base_lr = 0.0001
    rmsprop=keras.optimizers.Adam(lr=base_lr)
    model_centerloss.compile(optimizer=rmsprop, loss=["sparse_categorical_crossentropy", lambda y_true,y_pred: y_pred],loss_weights=[1,lambda_c],metrics=['accuracy'])
else:
    base_lr = 0.0001
    rmsprop=keras.optimizers.Adam(lr=base_lr)
    model=Model(inputs = image_batch, outputs = emotion_out)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])
    



if isCenterloss:
    base_lr = 0.0001
    EPOCH_NB = 100
    BATCH_NB = len(train_list_data)/batch_size
    BATCH_NB_TEST = len(validation_list_data)/batch_size
    lr_schedule = [base_lr]*100 + [base_lr*0.1]*50 + [base_lr*0.01]*50
    LOSSES = []
    LOSSES_test = []
    LOSSES.append(model_centerloss.metrics)
    LOSSES_test.append(model_centerloss.metrics)
    start = time.clock()
    for epoch_nb in xrange(EPOCH_NB):
        if epoch_nb>1 and lr_schedule[epoch_nb] != lr_schedule[epoch_nb-1]:
            model.save('../task_emotion/model_center_loss_epoch_%d_weights.h5'%epoch_nb)
            K.get_session().run(model_centerloss.optimizer.lr.assign(lr_schedule[epoch_nb]))
        Losses=[]
        for batch_nb in tqdm(xrange(BATCH_NB), total=BATCH_NB):
            [Image_data_emotion, Labels_emotion] = gen_train_emotion.next()
            print np.where(Labels_emotion['emotion']==1)[1].reshape(1,-1).transpose().shape
            losses = model_centerloss.train_on_batch([Image_data_emotion, np.where(Labels_emotion['emotion']==1)[1]],[Labels_emotion['emotion'],np.random.rand(Image_data_emotion.shape[0],1)])
            Losses.append(losses)
            if epoch_nb == 0: print losses
        Losses_test=[]
        for batch_nb in tqdm(xrange(BATCH_NB_TEST)):
            [Image_data_emotion, Labels_emotion] = gen_valid_emotion.next()
            losses = model_centerloss.test_on_batch([Image_data_emotion, np.where(Labels_emotion['emotion']==1)[1]],[Labels_emotion['emotion'],np.random.rand(Image_data_emotion.shape[0],1)])
            Losses_test.append(losses)
        print 'Done for Epoch %d.'% epoch_nb
        print model_centerloss.metrics
        print np.array(Losses).mean(axis=0)
        print np.array(Losses_test).mean(axis=0)
        LOSSES.append(Losses)
        LOSSES_test.append(Losses_test)
        pickle.dump({'LOSSES':LOSSES, 'LOSSES_test':LOSSES_test} , open("../task_emotion/center_loss_historty.pickle", 'w'))
    end = time.clock()
    print 'total_emotion_cost_time %d.' %  (end-start)
    model.save('../task_emotion/model_center_loss_last_weights.h5')
else:
    base_lr = 0.0001
    EPOCH_NB = 100
    BATCH_NB = len(train_list_data)/batch_size
    BATCH_NB_TEST = len(validation_list_data)/batch_size
    lr_schedule = [base_lr]*100 + [base_lr*0.1]*50 + [base_lr*0.01]*50
    LOSSES = []
    LOSSES_test = []
    LOSSES.append(model.metrics)
    LOSSES_test.append(model.metrics)
    start = time.clock()
    for epoch_nb in xrange(EPOCH_NB):
        if epoch_nb>1 and lr_schedule[epoch_nb] != lr_schedule[epoch_nb-1]:
            model.save('../task_emotion/model_epoch_%d_weights.h5'%epoch_nb)
            K.get_session().run(model.optimizer.lr.assign(lr_schedule[epoch_nb]))
        Losses=[]
        for batch_nb in tqdm(xrange(BATCH_NB), total=BATCH_NB):
            [Image_data_emotion, Labels_emotion] = gen_train_emotion.next()
            losses = model.train_on_batch(Image_data_emotion, Labels_emotion['emotion'])
            Losses.append(losses)
            if epoch_nb == 0: print losses
        Losses_test=[]
        for batch_nb in tqdm(xrange(BATCH_NB_TEST)):
            [Image_data_emotion, Labels_emotion] = gen_valid_emotion.next()
            losses = model.test_on_batch(Image_data_emotion, Labels_emotion['emotion'])
            Losses_test.append(losses)
        print 'Done for Epoch %d.'% epoch_nb
        print model.metrics_names
        print np.array(Losses).mean(axis=0)
        print np.array(Losses_test).mean(axis=0)
        LOSSES.append(Losses)
        LOSSES_test.append(Losses_test)
        pickle.dump({'LOSSES':LOSSES, 'LOSSES_test':LOSSES_test} , open("../task_emotion/historty.pickle", 'w'))
    end = time.clock()
    print 'total_emotion_cost_time %d.' %  (end-start)
    model.save('../task_emotion/model_last_weights.h5')


    
