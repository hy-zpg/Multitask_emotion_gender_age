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
import time
import generate_data_landmark_gender as gd_emo
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
import keras.backend as kb
from keras.layers import Dense, GlobalAveragePooling2D
# from keras.applications.mobilenetv2 import MobileNetV2
from keras.utils import plot_model
from keras_vggface.vggface import VGGFace

#define the  global variable
w0,h0=256, 256
crop_size=224
batch_size = 32
num_outputs_landmark = 10
num_outputs_gender=2
num_outputs_smile=2
num_outputs_glass=2
num_outputs_pose=5

#difine the model
# def multi_task_face():
#     num_outputs = 2
#     model = SqueezeNet(include_top=True, weights='imagenet',
#                input_tensor=None, input_shape=None,
#                pooling=None,
#                classes=1000)
#     x = model.get_layer('drop9').output
#     x = Convolution2D(num_outputs, (1, 1), padding='valid', name='conv10')(x)
#     x = Activation('relu', name='relu_conv10')(x)
#     x = GlobalAveragePooling2D()(x)
#     out = Activation('softmax', name='loss')(x)
#     drop2_out = model.get_layer('fire2/concat').output
#     conv1a = Convolution2D(128, (3, 3), strides=(2,2), padding='valid', name='conv_1a')(drop2_out)
#     conv1a = Activation('relu', name='relu_conv1a')(conv1a)
#     conv1a = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='conv1a_pool')(conv1a)
#     drop5_out = model.get_layer('pool5').output
#     drop9_out = model.get_layer('drop9').output
#     merged_out = merge([drop9_out, conv1a, drop5_out], mode='concat', concat_axis=-1, name='merged_out')
#     conv_final = Convolution2D(128, (1, 1), padding='valid', name='conv_final')(merged_out)
#     conv_final = Activation('relu', name='relu_conv_final')(conv_final) 
#     drop9_flat = Flatten(name='flat')(conv_final)
#     model=Model(inputs=model.input,outputs=[drop9_out,drop9_flat])
#     plot_model(model,to_file='../task_mtfl/model_lower.png',show_shapes=True)
#     return model

# #define the intermedia output
# image_batch = Input(shape=(crop_size, crop_size, 3), name='in_t1')
# model = multi_task_face()
# common_feat = model(image_batch)

# #define the final output
# landmark_FC = Dense(128,name='landmark_FC', activation="relu",init='he_normal',W_regularizer=l2(0.00))(common_feat[1])
# landmark_out = Dense(num_outputs_landmark,name='landmark_out',activation='relu',init='he_normal', W_regularizer=l2(0.00))(landmark_FC)

# gender_conv = Convolution2D(num_outputs_gender, (1, 1), padding='valid', name='gender_conv')(common_feat[0])
# gender_conv = Activation('relu', name='relu_gender_conv')(gender_conv)
# gender_conv = GlobalAveragePooling2D()(gender_conv)
# gender_out = Activation('softmax', name='gender_out')(gender_conv)

# smile_FC = Dense(128,name='smile_FC', activation="relu",init='he_normal',W_regularizer=l2(0.00))(common_feat[1])
# smile_out = Dense(num_outputs_smile,name='smile_out',activation='softmax',init='he_normal', W_regularizer=l2(0.00))(smile_FC)

# glass_FC = Dense(128,name='glass_FC', activation="relu",init='he_normal',W_regularizer=l2(0.00))(common_feat[1])
# glass_out = Dense(num_outputs_glass,name='glass_out',activation='softmax',init='he_normal', W_regularizer=l2(0.00))(glass_FC)

# pose_conv = Convolution2D(num_outputs_pose, (1, 1), padding='valid', name='pose_conv')(common_feat[0])
# pose_conv = Activation('relu', name='relu_pose_conv')(pose_conv)
# pose_conv = GlobalAveragePooling2D()(pose_conv)
# pose_out = Activation('softmax', name='pose_out')(pose_conv)
# #establish the whole network
# model=Model(inputs = image_batch, 
#             outputs = [landmark_out, gender_out,  smile_out,  glass_out, pose_out])

base = VGGFace(include_top=False,model = 'resnet50',weights='vggface',input_shape=(224,224,3))
last_layer = base.get_layer('avg_pool').output
top_layer = Flatten(name='flatten')(last_layer)
landmark_FC = Dense(128,name='landmark_FC', activation="relu")(top_layer)
landmark_out = Dense(10,name='landmark_out',activation='relu')(landmark_FC)

gender_FC = Dense(128,name='gender_FC', activation="relu")(top_layer)
gender_out = Dense(2,activation='softmax', name='gender_out')(gender_FC)

smile_FC = Dense(128,name='smile_FC', activation="relu")(top_layer)
smile_out = Dense(2,name='smile_out',activation='softmax')(smile_FC)

glass_FC = Dense(128,name='glass_FC', activation="relu")(top_layer)
glass_out = Dense(num_outputs_glass,name='glass_out',activation='softmax')(glass_FC)


pose_FC = Dense(128,name='pose_FC', activation="relu")(top_layer)
pose_out = Dense(num_outputs_pose,name='pose_out',activation='softmax')(pose_FC)

model = Model(inputs=base.input, outputs=[landmark_out, gender_out,  smile_out,  glass_out, pose_out], name='MTFLinceptionv3')


plot_model(model,to_file='../task_mtfl/model_upper.png',show_shapes=True)
model.summary()

#prepare the input data
list_dat_train_emo = gd_emo.get_train_list()
list_dat_valid_emo = gd_emo.get_val_list()
gen_train_emo = gd_emo.generate_data(list_dat_train_emo, batch_size, h0, w0, crop_size, SHUFFLE=True, RANDOM_CROP=True)
gen_valid_emo = gd_emo.generate_data(list_dat_valid_emo, batch_size, h0, w0, crop_size, SHUFFLE=True, RANDOM_CROP=False)


#prepare the  parameter of network
base_lr = 0.0001
rmsprop=keras.optimizers.Adam(lr=base_lr)
def custom_mse_lm(y_true,y_pred):
    return kb.sign(kb.sum(kb.abs(y_true),axis=-1))*kb.sum(kb.square(tf.multiply((kb.sign(y_true)+1)*0.5, y_true-y_pred)),axis=-1)/kb.sum((kb.sign(y_true)+1)*0.5,axis=-1)

def custom_mse_pose(y_true,y_pred):
    return kb.sign(kb.sum(kb.abs(y_true),axis=-1))*losses.mean_squared_error(y_true,y_pred)
#model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])
model.compile(optimizer='rmsprop',
              loss={'landmark_out': custom_mse_lm, 'gender_out': 'categorical_crossentropy','smile_out': 'categorical_crossentropy','glass_out': 'categorical_crossentropy','pose_out': custom_mse_pose},
              metrics = ['acc'])
EPOCH_NB = 100
BATCH_NB = len(list_dat_train_emo)/batch_size
BATCH_NB_TEST =  len(list_dat_valid_emo)/batch_size
lr_schedule = [base_lr]*100 + [base_lr*0.1]*50 + [base_lr*0.01]*50

#obtain the evaluate loss 
LOSSES = []
LOSSES_test = []
LOSSES.append(model.metrics_names)
LOSSES_test.append(model.metrics_names)

start = time.clock()
for epoch_nb in xrange(EPOCH_NB):
    if epoch_nb>1 and lr_schedule[epoch_nb] != lr_schedule[epoch_nb-1]:
        model.save('../task_mtfl/model_epoch_%d_weights.h5'%epoch_nb)
        K.get_session().run(model.optimizer.lr.assign(lr_schedule[epoch_nb]))
    Losses=[]
    for batch_nb in tqdm(xrange(BATCH_NB), total=BATCH_NB):
        [Image_data_emo, Labels_emo] = gen_train_emo.next()
        #print Labels_emo['gender'].shape
        losses = model.train_on_batch( Image_data_emo,
                                      [Labels_emo['landmark'],Labels_emo['gender'],
                                       Labels_emo['smile'], Labels_emo['glass'], Labels_emo['pose']])
        Losses.append(losses)
        if epoch_nb == 0: print losses
    Losses_test=[]
    for batch_nb in tqdm(xrange(BATCH_NB_TEST)):
        [Image_data_emo, Labels_emo] = gen_valid_emo.next()
        losses = model.test_on_batch( Image_data_emo,
                                      [ Labels_emo['landmark'],Labels_emo['gender'],
                                       Labels_emo['smile'], Labels_emo['glass'],Labels_emo['pose']])
        Losses_test.append(losses)


    print 'Done for Epoch %d.'% epoch_nb
    print model.metrics_names
    print np.array(Losses).mean(axis=0)
    print np.array(Losses_test).mean(axis=0)
    LOSSES.append(Losses)
    LOSSES_test.append(Losses_test)
    pickle.dump({'LOSSES':LOSSES, 'LOSSES_test':LOSSES_test} , open("../task_mtfl/historty.pickle", 'w'))
end = time.clock()
print 'total_mtfl_cost_time %d.' %  (end-start)
model.save('../task_mtfl/model_last_weights{val_acc:.2f}.h5')

    
