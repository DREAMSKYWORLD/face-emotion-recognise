from keras.models import load_model
from PrepareDataset import *
import os

# 加载模型
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
emotion_model_path = '../model/fer2013_XCEPTION.42-0.67.hdf5'
face_model_path = '../model/cnnmodel.h5'
face_model = load_model(face_model_path)
emotion_model = load_model(emotion_model_path, compile=False)

emotion_img_size = (48,48)
face_img_size = (62,47)
emotion_num_class = 7


def photoface_gg(image_path):

    labels_dic = {}
    people = [person for person in os.listdir("../people/")]
    for i, person in enumerate(people):
        labels_dic[i] = person

    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img_gray.shape

    faces_coord = detect_face(img)
    if len(faces_coord):
        faces = normalize_faces(img, faces_coord)
        for i, face in enumerate(faces):
            face = face.reshape(1, 62, 47, 1)
            pred = face_model.predict_proba(face)
            result = np.argmax(pred[0])
            print(pred, result)
            name = labels_dic[result].capitalize()

            x, y, w, h = faces_coord[i]
            # 裁剪出脸部图像
            emotion_img = cv2.resize(img_gray[y:y + h, x:x + w], (48, 48))
            # 调用模型预测情绪
            emotion_result = predict_emotion(emotion_img, emotion_model)
            emotion = emotion_labels[int(np.argmax(emotion_result))]

            cv2.putText(img, '%s' % emotion, (x + 2, y + h - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 255), thickness=2, lineType=1)

            cv2.putText(img, name, (faces_coord[i][0], faces_coord[i][1] - 10),
                            cv2.FONT_HERSHEY_PLAIN, 2, (66, 53, 243), 2)
            draw_rectangle(img, faces_coord)

            # cv2.namedWindow('Face', 0)
            # cv2.resizeWindow('Face', width, height)
            # cv2.imshow('Face', img)
            # k = cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # if k & 0xFF == ord('q'):
            #     break
    img = cv2.cvtColor(img, cv2.cv2.COLOR_BGR2RGB)
    return img, emotion_result, result

# if __name__ = '__main__':
#     photoface_gg('D:/1.jpg')
