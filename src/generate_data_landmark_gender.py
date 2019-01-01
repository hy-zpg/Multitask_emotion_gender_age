from PIL import Image
import numpy as np
from keras.utils import np_utils
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

root_dir = '/home/yanhong/Downloads/next_step/Multitask_emotion_based/MTFL/'
train_src = root_dir + 'training.txt' # training annotations
val_src   = root_dir + 'testing.txt'  # testing annotations


#img_src   = root_dir + 'crop_image'  # root add of images

def get_list(src):
    list_dat=[]
    for line in open(src).readlines():
        items=line.split()
        #print len(items)
        #landmark = np.float64(items[1:11]) ## 5 double point(x,y)
        landmark = items[1:11]
        gender = int(int(items[11]) == 1) ## 1 for male, 0 for female
        smile =  int(int(items[12]) == 1) ## 1 for smiling, 0 for non smiling
        glass =  int(int(items[13]) == 1) ## 1 for wearing glasses, 0 for no glasses
        pose = int(int(items[14]))-1 ## 5 kinds of pose
        #print items[0]
        list_dat.append({'add': items[0], 
                         'landmark':landmark,
                         'gender': gender, 
                         'smile': smile,
                         'glass': glass,
                         'pose':pose} )
        #print len(list_dat)
    #list_dat[:]['pose'] = np_utils.to_categorical(list_dat[:]['pose'], 5)
    return list_dat


def get_train_list():
    return get_list(train_src)

def get_val_list():
    return get_list(val_src)


def generate_data(list_dat, batch_size, h0, w0, crop_size, SHUFFLE=True, RANDOM_CROP=True):
    batch_nb=-1
    batch_total=len(list_dat)/batch_size
    assert(h0>=crop_size)
    assert(w0>=crop_size)
    h_max=h0-crop_size
    w_max=w0-crop_size
    h_max_half=int(0.5*h_max)
    w_max_half=int(0.5*w_max)
    
    if SHUFFLE: random.shuffle(list_dat)
    while 1:
        batch_nb+=1
        if batch_nb==batch_total:
            if SHUFFLE: random.shuffle(list_dat)
            batch_nb = 0
        start_idx=batch_nb*batch_size
        Image_data=np.zeros((batch_size,crop_size,crop_size,3),dtype=np.float32)
        Label_landmark = np.zeros((batch_size,10),dtype=np.float32)
        Label_gender=np.zeros((batch_size,2), dtype=np.float32)
        Label_smile=np.zeros((batch_size,2), dtype=np.float32)
        Label_glass=np.zeros((batch_size,2), dtype=np.float32)
        Label_pose=np.zeros((batch_size,5),dtype=np.float32)
        for k in xrange(batch_size):
            dat = list_dat[start_idx+k]  
            if RANDOM_CROP:
                x_off=np.random.randint(w_max)
                y_off=np.random.randint(h_max)
            else:
                x_off=w_max_half
                y_off=h_max_half
            #print image.img_to_array(Image.open(root_dir+dat['add']).shape
            img=image.img_to_array(Image.open(root_dir+dat['add']).convert('RGB').resize((w0,h0)))
            Image_data[k] = img[y_off:y_off+crop_size, x_off:x_off+crop_size,:]
            Label_landmark[k] = dat['landmark']
            Label_gender[k, dat['gender']] = 1
            Label_smile[k, dat['smile']]   = 1
            Label_glass[k, dat['glass']]   = 1
            Label_pose[k,dat['pose']] = 1
        Image_data= preprocess_input(Image_data)
        targets={'landmark': Label_landmark,'gender': Label_gender, 'smile': Label_smile, 'glass': Label_glass,'pose':Label_pose}
        yield [Image_data, targets]

def de_preprocess_image(image_demeaned):
    return (image_demeaned+[103.939, 116.779, 123.68])[:,:,::-1].astype(np.uint8)


def show_image_label(Image_data, Labels):
    gender_list=['F', 'M']
    glass_list=['NG', 'G']
    smile_list=['NS', 'S']
    pose_list = ['left','left_profile','frontal','right','right_profile']
    from pylab import *
    for k in xrange(Image_data.shape[0]):
        figure(1)
        imshow(de_preprocess_image(Image_data[k]))
        #print Labels[Labels['pose'][k].argmax()
        print Labels['pose'][k]
        #print Labels['gender'][k].shape
        #print pose_list[Labels['pose'][k].argmax()
        title('%s,%s, %s, %s'% (
                Labels['landmark'][k],
                gender_list[Labels['gender'][k].argmax()],
                glass_list[Labels['glass'][k].argmax()],
                smile_list[Labels['smile'][k].argmax()]
                #pose_list[Labels['pose'][k].argmax()]
                ))
        plt.show()


if __name__ == '__main__':
    list_dat = get_train_list()
    #list_dat=get_val_list()
    dat_gen = generate_data(list_dat,3, 128, 128, 112 , SHUFFLE=True, RANDOM_CROP=True)
    #dat_gen = generate_data(list_dat, 4, 256, 256, 224 , SHUFFLE=True, RANDOM_CROP=False)
    Image_data, Labels=dat_gen.next()
    print Image_data.shape
    print Labels['landmark'].shape
    print Labels['gender'].shape
    #show_image_label(Image_data, Labels)
