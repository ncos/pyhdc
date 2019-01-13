#!/usr/bin/python3

import csv, random, numpy as np
import keras
import keras.backend as K
from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, load_img
import os, sys, argparse, shutil, signal, glob, time

def aee_sq(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-1)

def aee_abs(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def aee_rel(y_true, y_pred):
    GT = K.sqrt(K.sum(K.square(y_true), axis=-1)) + K.epsilon()
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)) / GT

def model(load, shape, checkpoint=None):
    """Return a model from file or to train on."""
    if load and checkpoint: return load_model(checkpoint)

    conv_layers1, conv_layers2, dense_layers = [24, 36, 48], [64, 64], [1164, 200, 50, 10]

    model = Sequential()
    model.add(Convolution2D(24, (5, 5), activation='relu', input_shape=shape))
    model.add(MaxPooling2D())
    for cl in conv_layers1:
        model.add(Convolution2D(cl, (5, 5), strides=(2, 2), activation='relu'))
        #model.add(MaxPooling2D())
        model.add(BatchNormalization())

    for cl in conv_layers2:
        model.add(Convolution2D(cl, (3, 3), strides=(1, 1), activation='relu'))
        model.add(BatchNormalization())
        #model.add(MaxPooling2D())

    model.add(Flatten())
    for dl in dense_layers:
        model.add(Dense(dl, activation='relu'))
        model.add(BatchNormalization())
        #model.add(Dropout(0.5))

    model.add(Dense(3, activation='linear'))
    model.compile(loss=aee_sq, optimizer=Adam(lr=0.001), metrics=[aee_abs, aee_rel])
    return model


def get_X_y(base_dir, X, y, X_val, y_val, rate=10):
    with open(os.path.join(base_dir, 'cam_vels_local_frame.txt')) as fin:
        allines = fin.readlines()
        for i, line in enumerate(allines):
            split_line = line.split(' ')
            img_path = os.path.join(base_dir, 'slices', split_line[0])
            vx = float(split_line[1])
            vy = float(split_line[2])
            vz = float(split_line[3])

            len2 = vx * vx + vy * vy + vz * vz
            if (len2 < 0.4):
                continue
            #l = sqrt(len2)

            vx /= 2.0
            vy /= 2.0
            vz /= 2.0

            if (i % rate == 0):
                X_val.append(img_path)
                y_val.append([vx, vy, vz])
            else:
                X.append(img_path)
                y.append([vx, vy, vz])
 

def process_image(path, velocity, augment):
    """Process and augment an image."""
    image = load_img(path)
    image = img_to_array(image)
    #image[:,:,0] = 0
    #image[:,:,2] = 0

    #print (image.shape)
 
    if augment:
        image = random_shift(image, 0, 0.2, 0, 1, 2)  # only vertical
        if random.random() < 0.5:
            image = flip_axis(image, 1)
            steering_angle = -steering_angle

    image = image.astype(np.float32)
    image = image / 255 - 0.5

    return image, velocity


class DataGenerator(keras.utils.Sequence):
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.y = y
        self.X = X
        self.shuffle = shuffle
        self.size = int(np.floor(len(self.X) / self.batch_size))
        self.on_epoch_end()

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        idx = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        batch_X, batch_y = [], []
        for k in idx:
            image, sa = process_image(self.X[k], self.y[k], augment=False)
            batch_X.append(image)
            batch_y.append(sa)

        return np.array(batch_X), np.array(batch_y)

    def on_epoch_end(self):
        print ("Shuffling data")
        self.indexes = np.arange(len(self.X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)



def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--big', action='store_true')
    parser.add_argument('--batch',
                        type=int,
                        default=32,
                        required=False)
    args = parser.parse_args()




    """Load our network and our data, fit the model, save it."""
    net = model(load=False, shape=(260, 346, 3))

    X = []
    y = []
    X_val = []
    y_val = []

    #get_X_y("/home/ncos/pyhdc/test_data", X, y)
    get_X_y("/home/ncos/raid/EV-IMO/SET4/O1O2O3_0", X, y, X_val, y_val)
    get_X_y("/home/ncos/raid/EV-IMO/SET4/O1O2O3_1", X, y, X_val, y_val)
    get_X_y("/home/ncos/raid/EV-IMO/SET4/O1O2O3_3", X, y, X_val, y_val)
    
    if (args.big):
        print ("Processing big dataset")
        get_X_y("/home/ncos/raid/EV-IMO/SET4/O1O2O3_2", X, y, X_val, y_val)
        get_X_y("/home/ncos/raid/EV-IMO/SET4/O1O2O3_FAST", X, y, X_val, y_val)
        get_X_y("/home/ncos/raid/EV-IMO/SET4/O1O2O3_PLAIN_WALL", X, y, X_val, y_val)
        get_X_y("/home/ncos/raid/EV-IMO/SET4/O1O2O3_PLAIN_WALL_II", X, y, X_val, y_val)
        get_X_y("/home/ncos/raid/EV-IMO/SET4/O1O2O3_S3", X, y, X_val, y_val)
        get_X_y("/home/ncos/raid/EV-IMO/SET4/O1O3_PLAIN_WALL_P3", X, y, X_val, y_val)
        #get_X_y("/home/ncos/raid/EV-IMO/SET4/O1O3_TOP", X, y, X_val, y_val)

    print ("Input dataset size:", len(X), "data points")
    print ("\t\t", len(X_val), "validation points")

    #tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    batch_size = args.batch
    per_epoch = max(len(X) // batch_size, 1)
    val_per_epoch = max(len(X_val) // batch_size, 1)

    #shutil.rmtree('./logs')
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_grads=False,
                            write_graph=True, write_images=True)

    train_gen = DataGenerator(X, y, batch_size, True)
    val_gen   = DataGenerator(X_val, y_val, batch_size, True)

    #net.fit_generator(_generator(batch_size, X, y), steps_per_epoch=per_epoch, epochs=100, verbose=1,
    #                  callbacks=[tensorboard], validation_data=_generator(batch_size, X_val, y_val),
    #                  validation_steps=val_per_epoch)


    net.fit_generator(generator=train_gen, steps_per_epoch=per_epoch, epochs=200, verbose=1,
                      callbacks=[tensorboard], validation_data=val_gen,
                      validation_steps=val_per_epoch)

    net.save('model.h5')

if __name__ == '__main__':
    train()
