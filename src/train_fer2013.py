from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger,EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
from src.data_load import load_fer2013,split_data,preprocess_input
from src.train_fer2013_monitor import TrainingMonitor
import CNN
import matplotlib.pyplot as plt
from keras import optimizers
from keras.models import load_model

batch_size =32           #批数据大小
input_size =(48,48,1)      #输入图片尺寸
num_classes = 7            #7种情绪分类
patience = 50             #监测数据50轮训练不提升，停止训练
data_name = 'fer2013'
validation_split =0.2
num_epochs = 200
# sgd = optimizers.SGD(lr=0.001)

# 神经网络模型建立、配置
# model = CNN.mini_XCEPTION(input_size,num_classes)
# model = CNN.CNN_model(input_size,num_classes)
model = load_model('./train_fer2013_models/quan-batch32-0.1/fer2013_XCEPTION.42-0.67.hdf5')
# model = CNN.quan_XCEPTION(input_size,num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

#当被监测的数量不再提升，则停止训练。
early_stopping = EarlyStopping(monitor='val_loss',patience=patience,verbose=1)
#标准评估停止提升，降低学习速率
reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=12,verbose=1,factor=0.1)
#训练轮次结果数据保存
log_file = './train_fer2013_models/train_fer2013.log'
csv_logger = CSVLogger(log_file, append=False)
#训练模型保存
model_file = './train_fer2013_models/fer2013_XCEPTION.'+'{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_file,'val_loss',verbose=1,save_best_only=True)
#保存训练过程
train_monitor = TrainingMonitor('./train_fer2013_models/training_results.png')
#回调函数  callbacks
callbacks =[reduce_lr,early_stopping,csv_logger,model_checkpoint,train_monitor]

#处理训练数据
faces, emotions = load_fer2013()
#归一化处理
faces = preprocess_input(faces)
#训练集与测试集
train_data, val_data = split_data(faces, emotions, validation_split)
train_faces, train_emotions = train_data
# print(len(train_faces))

#实时数据增强
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=0.1,
                        horizontal_flip=True)

history=model.fit_generator(data_generator.flow(train_faces, train_emotions,
                                        batch_size),
                    steps_per_epoch=len(train_faces) / batch_size,
                    epochs=num_epochs, verbose=1, callbacks=callbacks,
                    validation_data=val_data)

# 绘制训练 & 验证的准确率值
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()




