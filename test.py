import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import numpy as np
import cv2
import os

labels = ['kyc', 'kjh']

x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

# 데이터셋 경로
# 얼굴 등록시 사진 넣을 폴더(test_dir, train_dir)
base_dir = 'C:\\Users\\user\\Desktop\\face_recognization'
train_dir = os.path.join(base_dir, 'train_data')
test_dir = os.path.join(base_dir, 'val_data')

# 하이퍼파라미터
batch_size = 32
epochs = 10
input_shape = (224, 224, 3)

# 데이터셋 불러오기 및 전처리
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)
print(train_generator.classes)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

def set_model():
    # 모델 구현
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(train_generator.num_classes, activation='softmax')
    ])

    # 모델 컴파일
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# 모델 학습 함수
def train(model):
    model.fit(train_generator, epochs=epochs, validation_data=test_generator)
    model.save("my_model")

# 학습시킨 모델 재사용하기
def get_model(name):
    return load_model(name)

# # 모델 새로 만들때
#  model = set_model()

# # 모델 학습
# train(model)

# 기존 모델로 할때
model = get_model("my_model")



# 모델 평가
test_loss, test_acc = model.evaluate(test_generator)
print(test_generator.classes)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# 이미지 전처리 함수
def preprocess_image(img_path):
    input_shape = (224, 224)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_shape)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# 이미지 분류 함수
def predict_image(img_path):
    img = preprocess_image(img_path)
    pred = model.predict(img)[0]
    idx = np.argmax(pred)
    return labels[idx]

# 입력 이미지 경로
img_path = "C:\\Users\\user\\Desktop\\face_recognization\\KakaoTalk_20230322_155433294.jpg"

# 이미지 분류 결과 출력
result = predict_image(img_path)
print('The input image is classified as:', result)

