import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import load_img,ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout,BatchNormalization,Activation
from tensorflow.keras.optimizers import Adam
import json

def plot_eg(plt):
    img_size = 48
    plt.figure(figsize=(12,20))
    ctr = 0

    for ex in os.listdir("/home/sreekesh/python/VS CODE/Face_recognition/Facial_Expression_Recognition/train/"):
        for i in range (1,6):
            ctr += 1
            plt.subplot(7,5,ctr)
            img = load_img("/home/sreekesh/python/VS CODE/Face_recognition/Facial_Expression_Recognition/train/" + ex +"/" + os.listdir("/home/sreekesh/python/VS CODE/Face_recognition/Facial_Expression_Recognition/train/" + ex )[1],target_size=(48,48))
            plt.imshow(img,cmap = "gray")

    plt.tight_layout()
    return plt


plot_eg(plt)    

for ex in os.listdir("/home/sreekesh/python/VS CODE/Face_recognition/Facial_Expression_Recognition/train/"):
    print(str(len(os.listdir("/home/sreekesh/python/VS CODE/Face_recognition/Facial_Expression_Recognition/train/"))) + " " + ex + " images")



data = ImageDataGenerator(horizontal_flip=True)
train_data = data.flow_from_directory("/home/sreekesh/python/VS CODE/Face_recognition/Facial_Expression_Recognition/train/",
                                        target_size=(48,48),
                                        color_mode="grayscale",
                                        batch_size= 64,
                                        class_mode="categorical",
                                        shuffle = True)

vali_data = data.flow_from_directory("/home/sreekesh/python/VS CODE/Face_recognition/Facial_Expression_Recognition/test/",
                                        target_size=(48,48),
                                        color_mode ="grayscale",
                                        class_mode="categorical",
                                        batch_size=64) 


model = Sequential()

model.add(Conv2D(64,(2,2),padding ="same",input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(256,(2,2),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(512,(2,2),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(1024,(2,2),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(1204))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(7,activation='softmax'))

opt = Adam(lr = 0.005)

model.compile(optimizer=opt,loss="categorical_crossentropy")

epochs = 10
steps_per_epoch = train_data.n // 64
validation_steps = vali_data.n // 64

history = model.fit(x=train_data,
                        steps_per_epoch=steps_per_epoch,
                        epochs= epochs,
                        validation_data = vali_data,
                        validation_steps=validation_steps)

model.save("/home/sreekesh/python/VS CODE/Face_recognition/Facial_Expression_Recognition/new_c_model.h5")

model_json = model.to_json()
with open("/home/sreekesh/python/VS CODE/Face_recognition/Facial_Expression_Recognition/new_c_model.json","w") as json_file :
    json_file.write(model_json)