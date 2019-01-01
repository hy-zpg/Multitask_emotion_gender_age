from PIL import Image
import numpy as np
import random
from keras.preprocessing.image import ImageDataGenerator,random_rotation,random_shift,random_shear,random_zoom
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import cv2
import os




detection_model_path = '/home/yanhong/Downloads/next_step/Xception/trained_models/detection_models/haarcascade_frontalface_default.xml'
detect_face_model = cv2.CascadeClassifier(detection_model_path)

#first step: detect the face and circle it

def format_image(image,crop_size,SIZE_FACE=48):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    gray_border = np.zeros((150, 150), np.uint8)
    gray_border[:, :] = 200
    gray_border[
        int((150 / 2) - (SIZE_FACE / 2)): int((150 / 2) + (SIZE_FACE / 2)),
        int((150 / 2) - (SIZE_FACE / 2)): int((150 / 2) + (SIZE_FACE / 2))
    ] = image
    image = gray_border
    faces = detect_face_model.detectMultiScale(
        image,
        scaleFactor=1.3,
        minNeighbors=5
    )
    if not len(faces) > 0:
        return None
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    face = max_area_face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
    try:
        image = cv2.resize(image, (crop_size, crop_size),
                           interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("[+] Problem during resize")
        return None
    return image


#second step: normalization to (0,1)
def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x
def depreprocess_input(x,v2=True):
    x=x/2.0
    x=x+0.5
    x=np.uint8(x*255)
    return x

#next series of processing
rotation_range=10
width_shift_range=0.1
height_shift_range=0.1
shear_range=0.08
zoom_range=0.1
def  series_process(img,rotation_range,width_shift_range,height_shift_range,shear_range,zoom_range):
    rotation_img=random_rotation(img, rotation_range,row_axis=0, col_axis=1, channel_axis=2 )
    shift_img = random_shift(rotation_img,width_shift_range, height_shift_range,row_axis=0, col_axis=1, channel_axis=2)
    shear_img = random_shear(shift_img,shear_range,row_axis=0, col_axis=1, channel_axis=2)
    #zoom_img = random_zoom(shear_range[(zoom_range,zoom_range)],row_axis=0, col_axis=1, channel_axis=2)
    return shear_img


img_file = '/home/yanhong/Downloads/next_step/Xception/datasets/ferplus/ferplus_images/'
label_file = '/home/yanhong/Downloads/next_step/Xception/datasets/ferplus/labels_list/'



def get_list(label_file):
    list_dat=[]
    alllines = []
    for i in range(8):
        fname = (label_file +'train-aug-c%d.txt' % (i))
        f = open(fname)
        lines = f.readlines()
        f.close()
        for line in lines:
            items = line.split()
            list_dat.append({'add':items[0].split('-')[0],
                'emotion':items[1]})
    print len(list_dat)
    fname = (label_file+ 'validation.txt')
    f=open(fname)
    lines = f.readlines()
    f.close()
    for lines in lines:
        items = line.split()
        list_dat.append({'add':items[0].split('-')[0],
            'emotion':items[1]})
    print len(list_dat)
    return list_dat


def split_train_validation(label_file,split=0.8):
    list_dat = get_list(label_file)
    train_length = int(len(list_dat)*split)
    train_list_dat = list_dat[:train_length]
    validation_list_data = list_dat[train_length:]
    return train_list_dat,validation_list_data


def generate_data(img_file,list_dat, batch_size, crop_size, SHUFFLE=True, TRAINING_PHASE=True):
    batch_nb=-1
    batch_total=len(list_dat)/batch_size
    if SHUFFLE: random.shuffle(list_dat)
    while 1:
        batch_nb+=1
        if batch_nb==batch_total:
            if SHUFFLE: random.shuffle(list_dat)
            batch_nb = 0
        start_idx=batch_nb*batch_size
        Image_data=np.zeros((batch_size,crop_size,crop_size,1),dtype=np.float32)
        Label_gen=np.zeros((batch_size,8), dtype=np.float32)
        for k in xrange(batch_size):
            dat = list_dat[start_idx+k]   
            img_path = img_file+dat['add'] + '.jpg'
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.resize(img, (64,64))
                #cv2.imshow('origin_img', img)
                #cv2.waitKey(500)
                #img = format_image(img,crop_size)
                #if img!=None:
                cv2.imshow('face_img',img)
                cv2.waitKey(400)
                print 'origin_img:'
                print img.shape,img[0][0:5]
                img = preprocess_input(img)
                #img = np.expand_dims(img, -1)
                #img = series_process(img,rotation_range,width_shift_range,height_shift_range,shear_range,zoom_range)
                print 'preprocessed_img:'
                print img.shape,img[0][0:5]
                Label_emotion=dat['emotion']
                Label_gen[k][Label_emotion] = 1
            else:
                continue
        #Image_data= preprocess_input(Image_data)
        targets = {'emotion':Label_gen}
        yield [Image_data, targets]

def de_preprocess_image(image_demeaned):
    return (image_demeaned+[103.939, 116.779, 123.68])[:,:,::-1].astype(np.uint8)


def show_image_label(Image_data, Labels):
    emotion_list=['0','1','2','3','4','5','6','7',]
    from pylab import *
    for k in xrange(Image_data.shape[0]):
        #img = np.squeeze(Image_data[k])
        #print 'squeezed:'
        img = depreprocess_input(Image_data[k],v2=True)
        print img.shape,img[0][0:10]
        #Image_data[k] = np.reshape(Image_data[k],(48,48))
        #Image_data[k] = np.reshape(Image_data[k],(48,48))
        #img =Image.fromarray(np.uint8(img*255))
        #img=cv2.fromarray(img*255)
        #img = np.uint8(img*255)
        cv2.imshow('restore_img',img)
        cv2.waitKey(1000)
        print Labels['emotion'][k].argmax()


if __name__ == '__main__':
    train_list_dat,validation_list_data = split_train_validation(label_file,split=0.8)
    dat_gen = generate_data(img_file,train_list_dat, 4, 48, SHUFFLE=True, TRAINING_PHASE=True)
    Image_data,Labels = dat_gen.next()
    show_image_label(Image_data,Labels)
    
