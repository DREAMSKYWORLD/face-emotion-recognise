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

    return labels_dic

labels_dic = collect_dataset()


# 加载模型
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
emotion_model_path = '../model/fer2013_XCEPTION.42-0.67.hdf5'
face_model_path = '../model/cnnmodel.h5'
face_model = load_model(face_model_path)
emotion_model = load_model(emotion_model_path, compile=False)

emotion_img_size = (48,48)
face_img_size = (62,47)
emotion_num_class = 7

def creat_video():
    cam = cv2.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces_coord = detect_face(frame)
        if len(faces_coord):
            faces = normalize_faces(frame, faces_coord)

            for i, face in enumerate(faces):
                face_img = face.reshape(1, 62, 47, 1)
                pred = face_model.predict_proba(face_img)
                print(pred)
                face_result = np.argmax(pred[0])
                # print(face_result)
                # print(labels_dic)
                name = labels_dic[face_result].capitalize()

                x,y,w,h = faces_coord[i]
                # 裁剪出脸部图像
                emotion_img = cv2.resize(gray_image[y:y + h, x:x + w], (48, 48))
                # 调用模型预测情绪
                emotion_result = predict_emotion(emotion_img, emotion_model)
                emotion = emotion_labels[int(np.argmax(emotion_result))]

                cv2.putText(frame, '%s' % emotion, (x + 2, y + h - 2), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 255), thickness=2, lineType=1)

                cv2.putText(frame, name, (faces_coord[i][0], faces_coord[i][1] - 10),
                            cv2.FONT_HERSHEY_PLAIN, 2, (66, 53, 243), 2)

            draw_rectangle(frame, faces_coord)

        cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2,
                    cv2.LINE_AA)

        cv2.imshow("opencv_face", frame)
        if cv2.waitKey(5) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    creat_video()