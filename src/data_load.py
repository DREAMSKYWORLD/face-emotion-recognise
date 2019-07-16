import pandas as pd
import numpy as np
import cv2
data_path = './data/fer2013.csv'
img_size = (48,48)

def load_fer2013():
    #读取fer2013.csv内容
    data = pd.read_csv(data_path)
    #提取pixels标签内容，转化为链表
    pixels = data['pixels'].tolist()
    width, height = 48,48
    faces = []
    #将链表pixels中的图片数据转化为48*48的数组后 添加到faces链表中
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), img_size)
        faces.append(face.astype('float32'))
    #将链表faces转化为数组 维度(None,48,48)
    faces = np.asarray(faces)
    #将faces维度变为(None,48,48,1)
    faces = np.expand_dims(faces, -1)
    #将emotion标签向量化
    emotions = pd.get_dummies(data['emotion']).values
    return faces, emotions

#划分训练集与测试集，前80%为训练集，后20%为测试集
def split_data(x, y, validation_split):
    num_samples = len(x)
    num_train_samples = int((1 - validation_split) * num_samples)
    train_x = x[:num_train_samples]
    train_y = y[:num_train_samples]
    val_x = x[num_train_samples:]
    val_y = y[num_train_samples:]
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data

def preprocess_input(x):
    x = x.astype('float32')
    x = x / 255.0
    x = x - 0.5
    x = x * 2.0
    return x