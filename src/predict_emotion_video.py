import cv2
import numpy as np
from keras.models import load_model

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
detection_model_path = '../xml/haarcascade_frontalface_default.xml'
emotion_model_path = '../model/fer2013_XCEPTION.42-0.67.hdf5'
img_size = 48
num_class = 7

# 使用opencv的人脸分类器
face_detection = cv2.CascadeClassifier(detection_model_path)

#读取训练好的模型
emotion_classifier = load_model(emotion_model_path, compile=False)


# 创建VideoCapture对象
cv2.namedWindow('Face')
video_capture = cv2.VideoCapture(0)

while True:

    bgr_image = video_capture.read()[1]
    #灰度化处理
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    #检测人脸位置
    facesLands = face_detection.detectMultiScale(gray_image,scaleFactor=1.1,
                                         minNeighbors=1, minSize=(120, 120))
    if len(facesLands) > 0:
        for faceLand in facesLands:
            x, y, w, h = faceLand
            images = []
            result = np.array([0.0] * num_class)

            # 裁剪出脸部图像
            image = cv2.resize(gray_image[y:y + h, x:x + w], (img_size, img_size))
            image = image * (1. / 255)
            image = image.reshape(1, img_size, img_size, 1)

            # 调用模型预测情绪
            predict_lists = emotion_classifier.predict(image)
            result += np.array([predict for predict_list in predict_lists
                                for predict in predict_list])
            emotion = emotion_labels[int(np.argmax(result))]
            print("Emotion:", emotion)

            # 框出脸部并且写上标签
            cv2.rectangle(bgr_image, (x - 20, y - 20), (x + w + 20, y + h + 20),
                          (0, 255, 255), 2)
            cv2.putText(bgr_image, '%s' % emotion, (x, y - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, 30)
            cv2.imshow('Face', bgr_image)

        if cv2.waitKey(60) == ord('q'):
            break

# 释放摄像头并销毁所有窗口
video_capture.release()
cv2.destroyAllWindows()









