#!/usr/bin/python3

import csv, random, numpy as np
from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img
import os, sys, shutil, signal, glob, time

def model(load, shape, checkpoint=None):
    """Return a model from file or to train on."""
    if load and checkpoint: return load_model(checkpoint)

    conv_layers, dense_layers = [32, 32, 64, 128], [1024, 512]
    
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, activation='elu', input_shape=shape))
    model.add(MaxPooling2D())
    for cl in conv_layers:
        model.add(Convolution2D(cl, 3, 3, activation='elu'))
        model.add(MaxPooling2D())
    model.add(Flatten())
    for dl in dense_layers:
        model.add(Dense(dl, activation='elu'))
        model.add(Dropout(0.5))
    model.add(Dense(3, activation='linear'))
    model.compile(loss='mse', optimizer="adam")
    return model
    
def get_X_y(base_dir, X, y):
    with open(os.path.join(base_dir, 'cam_vels_local_frame.txt')) as fin:
        for line in fin.readlines():
            split_line = line.split(' ')
            img_path = os.path.join(base_dir, 'slices', split_line[0])
            vx = float(split_line[1])
            vy = float(split_line[2])
            vz = float(split_line[3])

            X.append(img_path)
            y.append([vx, vy, vz])
 

def process_image(path, velocity, augment):
    """Process and augment an image."""
    image = load_img(path)
    image = img_to_array(image)

    #print (image.shape)
 
    if augment:
        image = random_shift(image, 0, 0.2, 0, 1, 2)  # only vertical
        if random.random() < 0.5:
            image = flip_axis(image, 1)
            steering_angle = -steering_angle

    image = (image / 255. - .5).astype(np.float32)
    return image, velocity

def _generator(batch_size, X, y):
    """Generate batches of training data forever."""
    while 1:
        batch_X, batch_y = [], []
        for i in range(batch_size):
            sample_index = random.randint(0, len(X) - 1)
            image, sa = process_image(X[sample_index], y[sample_index], augment=False)
            batch_X.append(image)
            batch_y.append(sa)
        yield np.array(batch_X), np.array(batch_y)

def train():
    """Load our network and our data, fit the model, save it."""
    net = model(load=False, shape=(260, 346, 3))

    X = []
    y = []
    get_X_y("/home/ncos/pyhdc/test_data", X, y)
    print ("Input dataset size:", len(X), "data points")

    net.fit_generator(_generator(256, X, y), samples_per_epoch=20224, nb_epoch=2)
    net.save('checkpoints/short.h5')

if __name__ == '__main__':
    train()
