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
import random
from PIL import Image
import sys
import generate_data_lap as gd_lap
import generate_data_emo_gender as gd_emo

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

from keras.utils.vis_utils import plot_model 
from tqdm import tqdm
import pickle

# gd_lap generate the age
#gd_emo generate the gender/smile/glass



w0,h0=256, 256
#crop_size=224
crop_size=224
batch_size = 32
batch_size_task1 = batch_size
batch_size_task2 = batch_size

num_outputs_age=101
num_outputs_gender=2
num_outputs_smile=2
num_outputs_glass=2

def multi_task_face():
    num_outputs = 2
    print "Building SqueezeNet..."
    #model = SqueezeNet()
    model = SqueezeNet(include_top=True, weights='imagenet',
               input_tensor=None, input_shape=None,
               pooling=None,
               classes=1000)
    print "Building New Layers..."
    x = model.get_layer('drop9').output
    x = Convolution2D(num_outputs, (1, 1), padding='valid', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)

    drop2_out = model.get_layer('fire2/concat').output
    conv1a = Convolution2D(128, (3, 3), strides=(2,2), padding='valid', name='conv_1a')(drop2_out)
    conv1a = Activation('relu', name='relu_conv1a')(conv1a)
    conv1a = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='conv1a_pool')(conv1a)
    drop5_out = model.get_layer('pool5').output
    drop9_out = model.get_layer('drop9').output
    merged_out = merge([drop9_out, conv1a, drop5_out], mode='concat', concat_axis=-1, name='merged_out')
    #conv_final = Convolution2D(256, (1, 1), padding='valid', name='conv_final')(merged_out)
    conv_final = Convolution2D(128, (1, 1), padding='valid', name='conv_final')(merged_out)
    conv_final = Activation('relu', name='relu_conv_final')(conv_final) 
    drop9_flat = Flatten(name='flat')(conv_final)
    #conv_all = Convolution2D(192,(1,1),padding='valid',name = 'conv_all')(drop9_flat)
    model=Model(inputs=model.input,outputs=[drop9_out,drop9_flat])
    #model=Model(inputs=model.input,outputs=drop9_flat)
    plot_model(model,to_file='model_lower.png',show_shapes=True)
    print "Model has been generated"
    return model

image_batch_t1 = Input(shape=(crop_size, crop_size, 3), name='in_t1')
image_batch_t2 = Input(shape=(crop_size, crop_size, 3), name='in_t2')
model = multi_task_face()
common_feat_t1 = model(image_batch_t1)
common_feat_t2 = model(image_batch_t2)


#age_FC = Dense(512, name='age_FC', activation='relu', init='he_normal', W_regularizer=l2(0.00))(common_feat_t1)
#age_out = Dense(num_outputs_age, name='age_out', activation='sigmoid', init='he_normal', W_regularizer=l2(0.00))(age_FC)

#gender_FC = Dense(128,name='gender_FC', activation="relu",init='he_normal',W_regularizer=l2(0.00))(common_feat_t2)
#gender_out = Dense(num_outputs_gender,name='gender_out',activation='softmax',init='he_normal', W_regularizer=l2(0.00))(gender_FC)
age_conv = Convolution2D(num_outputs_age, (1, 1), padding='valid', name='age_conv')(common_feat_t1[0])
age_conv = Activation('relu', name='relu_age_conv')(age_conv)
age_conv = GlobalAveragePooling2D()(age_conv)
age_out = Activation('softmax', name='age_out')(age_conv)

gender_conv = Convolution2D(num_outputs_gender, (1, 1), padding='valid', name='gender_conv')(common_feat_t2[0])
gender_conv = Activation('relu', name='relu_gender_conv')(gender_conv)
gender_conv = GlobalAveragePooling2D()(gender_conv)
gender_out = Activation('softmax', name='gender_out')(gender_conv)

smile_FC = Dense(128,name='smile_FC', activation="relu",init='he_normal',W_regularizer=l2(0.00))(common_feat_t2[1])
smile_out = Dense(num_outputs_smile,name='smile_out',activation='softmax',init='he_normal', W_regularizer=l2(0.00))(smile_FC)

glass_FC = Dense(128,name='glass_FC', activation="relu",init='he_normal',W_regularizer=l2(0.00))(common_feat_t2[1])
glass_out = Dense(num_outputs_glass,name='glass_out',activation='softmax',init='he_normal', W_regularizer=l2(0.00))(glass_FC)

model=Model(input = [image_batch_t1, image_batch_t2], 
            output = [age_out, gender_out, smile_out, glass_out])
plot_model(model,to_file='model_upper.png',show_shapes=True)
model.summary()


base_lr = 0.0001
rmsprop=keras.optimizers.Adam(lr=base_lr)
model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])

list_dat_train_age = gd_lap.get_train_list()
list_dat_valid_age = gd_lap.get_val_list(cast_age_to_int=True)

list_dat_train_emo = gd_emo.get_train_list()
list_dat_valid_emo = gd_emo.get_val_list()

gen_train_age = gd_lap.generate_data(list_dat_train_age, batch_size_task1, h0, w0, crop_size, SHUFFLE=True, RANDOM_CROP=True)
gen_valid_age = gd_lap.generate_data(list_dat_valid_age, batch_size_task1, h0, w0, crop_size, SHUFFLE=True, RANDOM_CROP=False)

gen_train_emo = gd_emo.generate_data(list_dat_train_emo, batch_size_task2, h0, w0, crop_size, SHUFFLE=True, RANDOM_CROP=True)
gen_valid_emo = gd_emo.generate_data(list_dat_valid_emo, batch_size_task2, h0, w0, crop_size, SHUFFLE=True, RANDOM_CROP=False)


EPOCH_NB = 50
BATCH_NB = np.max((len(list_dat_train_age)/batch_size_task1, len(list_dat_train_emo)/batch_size_task2))
BATCH_NB_TEST = np.min((len(list_dat_valid_age)/batch_size_task1, len(list_dat_valid_emo)/batch_size_task2))
lr_schedule = [base_lr]*100 + [base_lr*0.1]*50 + [base_lr*0.01]*50


LOSSES = []
LOSSES_test = []
LOSSES.append(model.metrics_names)
LOSSES_test.append(model.metrics_names)
for epoch_nb in xrange(EPOCH_NB):
    if epoch_nb>1 and lr_schedule[epoch_nb] != lr_schedule[epoch_nb-1]:
        model.save('model_epoch_%d_weights.h5'%epoch_nb)
        K.get_session().run(model.optimizer.lr.assign(lr_schedule[epoch_nb]))
    Losses=[]
    for batch_nb in tqdm(xrange(BATCH_NB), total=BATCH_NB):
        [Image_data_age, Labels_age] = gen_train_age.next()
        [Image_data_emo, Labels_emo] = gen_train_emo.next()
        losses = model.train_on_batch([Image_data_age, Image_data_emo],
                                      [Labels_age['age'], Labels_emo['gender'],
                                       Labels_emo['smile'], Labels_emo['glass']])
        Losses.append(losses)
        if epoch_nb == 0: print losses
    Losses_test=[]
    for batch_nb in tqdm(xrange(BATCH_NB_TEST)):
        [Image_data_age, Labels_age] = gen_valid_age.next()
        [Image_data_emo, Labels_emo] = gen_valid_emo.next()
        losses = model.test_on_batch([Image_data_age, Image_data_emo],
                                      [Labels_age['age'], Labels_emo['gender'],
                                       Labels_emo['smile'], Labels_emo['glass']])
        Losses_test.append(losses)


    print 'Done for Epoch %d.'% epoch_nb
    print model.metrics_names
    print np.array(Losses).mean(axis=0)
    print np.array(Losses_test).mean(axis=0)
    LOSSES.append(Losses)
    LOSSES_test.append(Losses_test)
    pickle.dump({'LOSSES':LOSSES, 'LOSSES_test':LOSSES_test} , open("historty.pickle", 'w'))

model.save('model_last_weights.h5')

    
