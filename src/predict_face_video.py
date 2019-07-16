import cv2
from keras.models import load_model
from PrepareDataset import *
import numpy as np
import os

labels_dic = {}

# 读取目录中的姓名建立标签对应关系
def collect_dataset():
    '''
     @brief 提去相片信息，并贴上标签
    :return:
    '''
    people = [person for person in os.listdir("../people/")]
    counter = 0
    for i, person in enumerate(people):
        labels_dic[i] = person
        # for image in os.listdir("people/" + person):
        #     if image.endswith('.jpg'):
        #         images.append(cv2.imread("people/" + person + '/' + image, 0))
        #         labels.append(i)
        # counter += 1
    return labels_dic


labels_dic = collect_dataset()

# Predict
filepath = '../model/cnnmodel.h5'
model = load_model(filepath)

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
cv2.namedWindow("opencv_face ", cv2.WINDOW_AUTOSIZE)

while True:
    ret, frame = cam.read()

    faces_coord = detect_face(frame)  # detect more than one face
    # print(faces_coord, type(faces_coord))
    if len(faces_coord):
        faces = normalize_faces(frame, faces_coord)

        for i, face in enumerate(faces):  # for each detected face
            face = face.reshape(1, 62, 47, 1)
            pred = model.predict_proba(face)
            result = np.argmax(pred[0])
            print(pred, result)
            name = labels_dic[result].capitalize()
            print(name)

            cv2.putText(frame, name, (faces_coord[i][0], faces_coord[i][1] - 10),
                        cv2.FONT_HERSHEY_PLAIN, 2, (66, 53, 243), 2)

        draw_rectangle(frame, faces_coord)  # rectangle around face

    cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2,
                cv2.LINE_AA)

    cv2.imshow("opencv_face", frame)  # live feed in external
    if cv2.waitKey(5) == 27:
        break

cam.release()
cv2.destroyAllWindows()
