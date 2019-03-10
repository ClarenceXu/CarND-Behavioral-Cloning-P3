import csv
import cv2
import os
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout
from keras.layers.convolutional import Conv2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from math import ceil


# for the sake of easy implementation
# I merged all IMG of different data source to single IMG via script get_data.sh
img_data_path = '/opt/data/IMG/'
csv_data_path = ['/opt/data/own_data/data/driving_log.csv',
                 '/opt/data/more_data/data/driving_log.csv',
                 '/opt/data/udacity_data/data/driving_log.csv']


def get_samples():
    samples = []
    for file in csv_data_path:
        with open(file) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                samples.append(line)
    return samples


# origin image 160x320x3
# Manually crop unrelavant part: [0-70],[150-160]
# Nvidia Model input: 66x200x3
# convert to YUV
def process_image(img_path):
    # trim image
    img = cv2.imread(img_path)
    # 80 x 320
    crop_img = img[70:150, 0:320]
    resized_img = cv2.resize(crop_img, (200, 66), interpolation=cv2.INTER_AREA)
    yuv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2YUV)
    return yuv_img


def generator(samples, batch_size=32):
    while 1:  # Loop forever so the generator never terminates
        num_samples = len(samples)
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])
                # create adjusted steering measurements for the side camera images
                correction = 0.25  # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                # read in images from center, left and right cameras
                img_center = process_image(img_data_path + os.path.basename(batch_sample[0]))
                img_left = process_image(img_data_path + os.path.basename(batch_sample[1]))
                img_right = process_image(img_data_path + os.path.basename(batch_sample[2]))
                # add images and angles to data set
                images.extend([img_center, img_left, img_right])
                angles.extend([steering_center, steering_left, steering_right])
                images.extend([cv2.flip(img_center, 1), cv2.flip(img_left, 1), cv2.flip(img_right, 1)])
                angles.extend([steering_center * -1.0, steering_left * -1.0, steering_right * -1.0])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def get_nvidia_model():
    model = Sequential()

    # Normalize, output 3@66x200
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)))
    # output 24@31x98
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
    # output 36@14x47
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
    # output 48@5x22
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
    # output 64@3x20
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    # output 64@1x18
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()
    model.compile(optimizer='Adam', loss='mse')  #loss='categorical_crossentropy'
    return model


def train_model(samples, model, model_name):
    batch_size = 128
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

    checkpoint = ModelCheckpoint(model_name + '-batch:{}-'.format(batch_size) + '{epoch:02d}.h5', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    history_object = model.fit_generator(train_generator,
                                         steps_per_epoch=ceil(len(train_samples) / batch_size),
                                         validation_data=validation_generator,
                                         validation_steps=ceil(len(validation_samples) / batch_size),
                                         epochs=50, verbose=1, callbacks=[checkpoint,early_stopping])
    return history_object


def print_model(history_object):
    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()



## ---- main function ---- ##


model_name = 'model'
samples = get_samples()
print('total sample size: ', len(samples))
model = get_nvidia_model()
history_object = train_model(samples, model, model_name)
model.save(model_name + '.h5')
print('model saved')
print_model(history_object)




# model-1 : epicos 20, batch 32 # work / fails
# model-2 : epicos 20, batch 64 # fails
# model-3 : epicos 20, batch 96 # fail
# model-4 : epicos 20, batch 128
# model-4 : epicos 30, batch 128 # fails
# model-5 : epicos 50, batch 128 # fails
# model-6 : epicos 50, batch 64 # fails
# model-7 : epicos 50, batch 96 # lowest loss, but doesn't drive through
# model-8 : epicos 50, batch 96 # more_data , fails
# model-8 : epicos 50, batch 128 # more_data ,
# model-9 : epicos 20, batch 32 # more_data ,
# model-a : epicos 20, batch 64 # more_data ,
# model-b : epicos 20, batch 96 # more_data ,
# model-c : epicos 50, batch 128 # more_data ,
# model-d : epicos 10, batch 128 # more_data ,