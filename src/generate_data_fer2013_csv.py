from PIL import Image
import numpy as np
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import pandas as pd
import cv2
from pylab import *
import time
import matplotlib.pyplot as plt 


root_dir = '/home/yanhong/Downloads/next_step/Xception/datasets/fer2013/fer2013.csv'


def  tuple_data(src):
       data = pd.read_csv(src)
       pixels = data['pixels'].tolist()
       width,height = 48, 48
       faces = []
       for pixel_sequence in pixels:
              face = [int(pixel) for pixel in pixel_sequence.split(' ')]
              face = np.asarray(face).reshape(width,height)
              #print face.shape
              #face = cv2.resize(face.astype('uint8'), (crop_size,crop_size),interpolation=cv2.INTER_AREA)
              #face = cv2.cvtColor(face,cv2.COLOR_GRAY2RGB)
              #faces.append(face.astype('float32'))
              faces.append(face)
       #faces = np.asarray(faces)
       emotions = pd.get_dummies(data['emotion']).as_matrix()
       return faces,emotions

#split the train and test
def split_train_validation_data(src,split_size):
       faces,emotions = tuple_data(src)
       data_length = len(faces)
       train_faces = faces[: int(split_size*data_length)]
       train_emotions = emotions[:int(split_size*data_length)]
       validation_faces = faces[int(split_size*data_length):]
       validation_emotions = emotions[int(split_size*data_length):]
       return train_faces,train_emotions,validation_faces,validation_emotions


#generate the iterate data
def generate_data(faces,emotions,crop_size,batch_size):
       batch_nb=-1
       batch_total = len(faces)/batch_size
       while 1:
              batch_nb+=1
              if batch_nb == batch_total:
                     batch_nb = 0
              start_idx = batch_nb * batch_size
              Image_data = np.zeros((batch_size,crop_size,crop_size,3),dtype=np.float32)
              Label_gen=np.zeros((batch_size,7), dtype=np.float32)
              print 'start generating the data_gen'
              for i in xrange(batch_size):
                     #print faces[start_idx+i].shape
                     plt.show(faces[start_idx+i])
                     faces[start_idx+i] = cv2.resize(faces[start_idx+i].astype('uint8'), (crop_size,crop_size),interpolation=cv2.INTER_AREA)
                     #print faces[start_idx+i].shape
                     plt.show(faces[start_idx+i])
                     faces[start_idx+i] = cv2.cvtColor(faces[start_idx+i],cv2.COLOR_GRAY2RGB)
                     #print faces[start_idx+i].shape
                     plt.show(faces[start_idx+i])
                     faces[start_idx+i] = np.asarray(faces[start_idx+i])
                     Image_data[i]= faces[start_idx+i]
                     Label_gen[i] = emotions[start_idx+i]
              Image_data = preprocess_input(Image_data)
              #print Image.shape
              #print label.shape
              #print 'end up generating data_gen'
              yield [Image_data,Label_gen]

def de_preprocess_image(image_demeaned):
    return (image_demeaned+[103.939, 116.779, 123.68])[:,:,::-1].astype(np.uint8)

def show_image_label(Image_data, Labels):
    for k in xrange(Image_data.shape[0]):
        print k,Image_data.shape
        figure(1)
        imshow(de_preprocess_image(Image_data[k]))
        title('emotion: %d'% (Labels[k].argmax()))
        plt.show()

if __name__ == '__main__':
    start = time.clock()
    train_faces,train_emotions,validation_faces,validation_emotions = split_train_validation_data(root_dir,0.8)
    end = time.clock()
    print 'total time is %d' % (end-start)
    train_dat_gen = generate_data(train_faces,train_emotions,224,4)
    Image_data, Labels=train_dat_gen.next()
    show_image_label(Image_data, Labels)
