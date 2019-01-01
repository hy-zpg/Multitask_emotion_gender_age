from PIL import Image
import numpy as np
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import cv2

root_dir = '/home/yanhong/Downloads/next_step/Multitask_emotion_based/fer3013/'
csv_file = '/home/yanhong/Downloads/next_step/Multitask_emotion_based/fer3013/file.csv'

def get_list(src):
    list_dat=[]
    for line in open(src).readlines():
        items=line.split(',')
        #print items[0]
        list_dat.append({'add': items[0], 
                         'emotion': int(items[1])-1})
    return list_dat


def split_train_validation(src,split=0.8):
    list_dat = get_list(src)
    train_length = int(len(list_dat)*split)
    train_list_dat = list_dat[:train_length]
    validation_list_data = list_dat[train_length:]
    return train_list_dat,validation_list_data


def generate_data(list_dat, batch_size, crop_size, SHUFFLE=True, TRAINING_PHASE=True):
    batch_nb=-1
    batch_total=len(list_dat)/batch_size
    if SHUFFLE: random.shuffle(list_dat)
    while 1:
        batch_nb+=1
        if batch_nb==batch_total:
            if SHUFFLE: random.shuffle(list_dat)
            batch_nb = 0
        start_idx=batch_nb*batch_size
        #Image_data=np.zeros((batch_size,crop_size,crop_size,3),dtype=np.float32)
        Image_data=np.zeros((batch_size,crop_size,crop_size,1),dtype=np.float32)
        Label_gen=np.zeros((batch_size,7), dtype=np.float32)
        for k in xrange(batch_size):
            dat = list_dat[start_idx+k]   
            #cv2.resize(faces[start_idx+i].astype('uint8'), (crop_size,crop_size),interpolation=cv2.INTER_AREA)
            #img=image.img_to_array(Image.open(root_dir+dat['add']).convert('RGB').resize( (crop_size,crop_size)))
            img=image.img_to_array(Image.open(root_dir+dat['add']).resize( (crop_size,crop_size)))
            Label_emotion=dat['emotion']
            Image_data[k] = img
            Label_gen[k][Label_emotion] = 1
        #Image_data= preprocess_input(Image_data)
        targets = {'emotion':Label_gen}
        yield [Image_data, targets]

def de_preprocess_image(image_demeaned):
    return (image_demeaned+[103.939, 116.779, 123.68])[:,:,::-1].astype(np.uint8)


def show_image_label(Image_data, Labels):
    emotion_list=['1','2','3','4','5','6','7',]
    from pylab import *
    for k in xrange(Image_data.shape[0]):
        print k,Image_data.shape
        figure(1)
        Image_data[k] = image.img_to_array(Image_data[k])
        print Image_data[k].shape
        #cv2.imshow('image',Image_data[k])
        #cv2.waitKey(1000)
        #imshow(de_preprocess_image(Image_data[k]))
        imshow(Image_data[k])
        print Labels['emotion'][k].argmax()
        #title('emotion: %d'% (Labels['emotion'][k].argmax()))
        plt.show()

'''if __name__ == '__main__':
    train_list_dat,validation_list_data = split_train_validation(csv_file,split=0.8)
    dat_gen = generate_data(train_list_dat,4)'''

if __name__ == '__main__':
    train_list_dat,validation_list_data = split_train_validation(csv_file,split=0.8)
    dat_gen = generate_data(train_list_dat, 4, 48, SHUFFLE=True, TRAINING_PHASE=True)
    Image_data,Labels = dat_gen.next()
    #image = image.array_to_image(Image_dataI)
    show_image_label(Image_data,Labels)
    
