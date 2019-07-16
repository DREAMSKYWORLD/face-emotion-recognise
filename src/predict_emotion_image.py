import sys
from keras.models import load_model
import cv2
import numpy as np

img_size = 48
num_class = 7
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
detection_model_path = './xml/frontal_face.xml'
emotion_model_path = './model/fer2013_XCEPTION.42-0.67.hdf5'
# emotion_model_path = './model/fer2013_CNN.62-0.66.hdf5'

#读取训练好的模型
model = load_model(emotion_model_path, compile=False)

def predict_emotion(face_img):
    image = cv2.resize(face_img, (img_size, img_size))
    image = image * (1. / 255)
    image = image.reshape(1, img_size, img_size, 1)

    # 调用模型预测情绪
    result=np.array([0.0] * num_class)
    predict_lists = model.predict(image)
    result += np.array([predict for predict_list in predict_lists
                        for predict in predict_list])
    return result

def face_detect(image_path):

    faceCasccade = cv2.CascadeClassifier(detection_model_path)

    # 读取图片信息
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #检测人脸位置
    faces = faceCasccade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(30, 30),
    )
    return faces, img_gray, img


if __name__ == '__main__':
    images = []
    flag = 0
    if len(sys.argv) == 2:
        images.append(sys.argv[1])
    else:
        print('there should be a picture after py')

    for image in images:
        faces, img_gray, img = face_detect(image)
        height,width = img_gray.shape
        face_exists = 0
        for (x, y, w, h) in faces:
            face_exists = 1
            face_img_gray = img_gray[y:y + h, x:x + w]
            results = predict_emotion(face_img_gray)  # face_img_gray

            angry, disgust, fear, happy, sad, surprise, neutral = results
            # 输出所有情绪的概率
            print('angry:   ', angry, '\ndisgust: ', disgust, '\nfear:    ', fear, '\nhappy:   ', happy, '\nsad:     ', sad,
                  '\nsurprise:', surprise, '\nneutral: ', neutral)
            # 输出最大概率的情绪
            index = np.argmax(results)
            emotion = emotion_labels[int(index)]
            print('Emotion :', emotion)

            #框出脸部并且写上标签
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.putText(img, emotion, (x + 2, y + h - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 255), thickness=2, lineType=1)
        #显示识别后的图片
        if face_exists:
            cv2.namedWindow('Face', 0)
            cv2.resizeWindow('Face',width,height)
            cv2.imshow('Face', img)
            k = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if k & 0xFF == ord('q'):
                break

