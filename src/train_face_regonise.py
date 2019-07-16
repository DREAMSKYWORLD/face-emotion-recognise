import numpy as np
from PrepareDataset import *
from AddNewFace import *
import cv2
import os
import random
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
# def cnn model
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dropout

def collect_dataset():
    '''
     @brief 提去相片信息，并贴上标签
    :return:
    '''
    people = [person for person in os.listdir("../people/")]
    counter = 0
    labels_dic = {}
    images = []
    labels = []
    for i, person in enumerate(people):
        labels_dic[i] = person
        for image in os.listdir("../people/" + person):
            if image.endswith('.jpg'):
                images.append(cv2.imread("../people/" + person + '/' + image, 0))
                labels.append(i)
        counter += 1
    return images, np.array(labels), labels_dic, counter

def train_model(name):
    images = []
    labels = []
    labels_dic = {}
    counter = 0

    # choice = input("Do you want to add new face? (Yes or No) ")
# if choice == 'yes' or choice == 'Yes' or choice == 'Y' or choice == 'y':
#     add_face(name)
    if name is not None:
        add_face(name)
    else:
        print('No Add!')


    images, labels, labels_dic, counter = collect_dataset()

    # X_train = np.asarray(images)
    # train = X_train.reshape(len(X_train), -1)
    train = np.asarray(images)
    X_train, X_test, Y_train, Y_test = train_test_split(train, labels, test_size=0.2, random_state=random.randint(0, 100))

    # 格式化和标准化
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = X_train.reshape(len(X_train), 62, 47, 1) / 255
    X_test = X_test.reshape(len(X_test), 62, 47, 1) / 255

    # print(X_train.shape[1:])
    # 将标签转化为二进制分类矩阵
    Y_train = np_utils.to_categorical(Y_train)
    Y_test = np_utils.to_categorical(Y_test)


    # 建立一个基于 CNN 的人脸识别模型
    model = Sequential()
    #建立一个CNN模型，一层卷积、一层池化、一层卷积、一层池化、抹平之后进行全链接、最后进行分类
    # 卷积层
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=X_train.shape[1:]))
    # 激活层
    model.add(Activation('relu'))
    # 最大池化层
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(
        Convolution2D(filters=64, kernel_size=(3, 3), padding='same')
    )
    model.add(Activation('relu'))
    model.add(
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
    )

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))

    model.add(Dense(counter))
    model.add(Activation('softmax'))
    model.summary()

    # training model
    model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
    # model.fit(X_train, Y_train, epochs=1, batch_size=50)
    history = model.fit(X_train, Y_train, validation_split=0.25, epochs=300, batch_size=32, verbose=1)
    filepath = '../model/cnnmodel.h5'

    print('Testing......')
    loss, accuracy = model.evaluate(X_test, Y_test)
    print('test loss: ', loss)
    print('test accuracy: ', accuracy)

    model.save(filepath)

    import matplotlib.pyplot as plt


    # Plot training & validation accuracy values
    plt.figure()
    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()

    # Plot training & validation loss values
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()