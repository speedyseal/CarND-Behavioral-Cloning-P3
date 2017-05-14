import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

# steering throttle, brake, speed

import csv
import cv2
import pandas as pd

def loadDriveLog(csvpath):
    csvfilename = csvpath + "driving_log.csv"
    lines = []
    with open(csvfilename) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

def loadImages(csvEntries, image_path):
    ### csvEntries - output of loadDriveLog
    images = []
    steering_angle = []
    for line in csvEntries:
        source_path = line[0]
        imgfilename = source_path.split('\\')[-1]
        imgpath = image_path + imgfilename
        images.append(cv2.imread(imgpath))
        steering_angle.append(float(line[3]))
    return images, steering_angle

def loadImage(source_path, image_path):
    imgfilename = source_path.split('\\')[-1]
    imgpath = image_path + imgfilename
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def loadImagesPd(df, image_path):
    it = df.itertuples(index=True)
    firstrow = next(it)
    img = loadImage(firstrow.center, image_path) # center image
    if firstrow.flip:
        img = cv2.flip(img, 1)

    images = np.zeros((df.shape[0], img.shape[0], img.shape[1], img.shape[2]), dtype=img.dtype)
    images[0] = img
    #images = [img]
    for i,row in enumerate(it):
        img= loadImage(row.center, image_path)   # center image
        if row.flip:
            img = cv2.flip(img, 1)
        images[i+1] = img
    #    images.append(img)
    # return np.array(images)
    return images
        
def plotimage(x):
    plt.imshow(x)
    plt.show()
    return x
    
def nvidiaModel():
    model = Sequential()
    model.add(Lambda(lambda x:(x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(((50,20), (0,0)), input_shape=(160,320,3)))
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    return model

def generate_data(df, image_path, batch_size):
    while True:
        shuffled_order = np.random.choice(df.shape[0], size=df.shape[0], replace=False)
        n_samples = df.shape[0]
        steps = int(np.ceil(n_samples/batch_size))
        for i in range(steps):
            starti = i*batch_size
            endi   = min(starti+batch_size, n_samples)
            idx = shuffled_order[starti:endi]
            # create Numpy arrays of input data
            # and labels, from each line in the file
            src_path = df.iloc[idx]
            x = loadImagesPd(src_path, image_path)
            y = df.iloc[idx]["angle"].values
            yield (x, y)

        
def train_gen(model, df, image_path, batch_size=32, epochs=5):
    train, validation = train_test_split(df, test_size = 0.2)
    model.compile(loss='mse', optimizer='adam')
    n_train = train.shape[0]
    history_object = model.fit_generator(generate_data(train, image_path, batch_size),
                                         samples_per_epoch=n_train, nb_epoch=epochs, 
                                         validation_data=generate_data(validation, image_path, batch_size),
                                         nb_val_samples=validation.shape[0],
                                         pickle_safe=True,
                                         nb_worker=4)

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
    return history_object

def train(model, X_train, y_train, epochs=5):
    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=epochs)
    
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
    return history_object

def main():
    model = nvidiaModel()
    model.save('model.h5')

if __name__ == "__main__":
    # execute only if run as a script
    main()
