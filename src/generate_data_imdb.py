from PIL import Image
import numpy as np
from keras.utils import np_utils
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from scipy.io import loadmat

root_dir = '/home/yanhong/Downloads/next_step/Xception/datasets/imdb_crop/'


#img_src   = root_dir + 'crop_image'  # root add of images

def get_list(root_dir):
    face_score_treshold = 3
    dataset_path = root_dir + 'imdb.mat'
    dataset = loadmat(dataset_path)
    image_names_array = dataset['imdb']['full_path'][0, 0][0]
    gender_classes = dataset['imdb']['gender'][0, 0][0]
    age_classes = dataset['imdb']['age'][0,0][0]
    face_score = dataset['imdb']['face_score'][0, 0][0]
    second_face_score = dataset['imdb']['second_face_score'][0, 0][0]
    face_score_mask = face_score > face_score_treshold
    second_face_score_mask = np.isnan(second_face_score)
    unknown_gender_mask = np.logical_not(np.isnan(gender_classes))
    unknown_age_mask = np.logical_not(np.isnan(age_classes))
    mask = np.logical_and(face_score_mask, second_face_score_mask)
    mask = np.logical_and(mask, unknown_gender_mask)
    mask = np.logical_and(mask,unknown_age_mask)
    image_names_array = image_names_array[mask]
    gender_classes = gender_classes[mask].tolist()
    age_classes = age_classes[mask].tolist()
    targets = [np.array(gender_classes),np.array(age_classes)]
    targets = np.transpose(targets)
    image_names = []
    for image_name_arg in range(image_names_array.shape[0]):
        image_name = image_names_array[image_name_arg][0]
        image_names.append(image_name)
    #return dict(zip(image_names, gender_classes)), dict(zip(image_names,age_classes))
    return dict(zip(image_names,targets))
   
def split_imdb_data(ground_truth_data, validation_split=.2, do_shuffle=False):
    ground_truth_keys = sorted(ground_truth_data.keys())
    if do_shuffle == True:
        shuffle(ground_truth_keys)
    training_split = 1 - validation_split
    num_train = int(training_split * len(ground_truth_keys))
    train_keys = ground_truth_keys[:num_train]
    validation_keys = ground_truth_keys[num_train:]
    return train_keys, validation_keys


if __name__ == '__main__':
    ground_truth_data = get_list(root_dir)
    train_keys, val_keys = split_imdb_data(ground_truth_data, 0.2)
    print len(train_keys)
    print train_keys[0]
    print ground_truth_data[train_keys[0]]
    print ground_truth_data[train_keys[0]][0]

