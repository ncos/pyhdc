#!/usr/bin/python3

import csv, random, numpy as np
import keras
import keras.backend as K
from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img
import os, sys, shutil, signal, glob, time


def aee_abs(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true)))

def aee_rel(y_true, y_pred):
    GT = K.sqrt(K.sum(K.square(y_true))) + K.epsilon()
    return K.sqrt(K.sum(K.square(y_pred - y_true))) / GT


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
    model.compile(loss='mse', optimizer="adam", metrics=[aee_abs, aee_rel])
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
    image[:,:,0] = 0
    image[:,:,2] = 0

    #print (image.shape)
 
    if augment:
        image = random_shift(image, 0, 0.2, 0, 1, 2)  # only vertical
        if random.random() < 0.5:
            image = flip_axis(image, 1)
            steering_angle = -steering_angle

    image = image.astype(np.float32)
    image = image / 255 - 0.5

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
    #get_X_y("/home/ncos/pyhdc/test_data", X, y)
    get_X_y("/home/ncos/raid/EV-IMO/SET4/O1O2O3_3", X, y)
    get_X_y("/home/ncos/raid/EV-IMO/SET4/O1O2O3_1", X, y)
    get_X_y("/home/ncos/raid/EV-IMO/SET4/O1O2O3_0", X, y)
    get_X_y("/home/ncos/raid/EV-IMO/SET4/O1O2O3_FAST", X, y)
    get_X_y("/home/ncos/raid/EV-IMO/SET4/O1O2O3_PLAIN_WALL", X, y)
    get_X_y("/home/ncos/raid/EV-IMO/SET4/O1O2O3_PLAIN_WALL_II", X, y)
    get_X_y("/home/ncos/raid/EV-IMO/SET4/O1O2O3_S3", X, y)
    get_X_y("/home/ncos/raid/EV-IMO/SET4/O1O3_PLAIN_WALL_P3", X, y)
    get_X_y("/home/ncos/raid/EV-IMO/SET4/O1O3_TOP", X, y)

    X_val = []
    y_val = []
    get_X_y("/home/ncos/raid/EV-IMO/SET4/O1O2O3_2", X_val, y_val)

    print ("Input dataset size:", len(X), "data points")

    #tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    batch_size = 32
    per_epoch = max(len(X) // batch_size, 1)
    val_per_epoch = max(len(X_val) // batch_size, 1)

    #shutil.rmtree('./logs')
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_grads=False,
                            write_graph=True, write_images=True)

    net.fit_generator(_generator(batch_size, X, y), steps_per_epoch=per_epoch, epochs=100, verbose=1,
                      callbacks=[tensorboard], validation_data=_generator(batch_size, X_val, y_val),
                      validation_steps=val_per_epoch)
    net.save('model.h5')

if __name__ == '__main__':
    train()
