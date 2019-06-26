import pandas as pd
import numpy as np
import cv2
from skimage.io import imread 
import os
import csv

from PIL import Image

import argparse
import numpy as np 
import scipy.misc

import matplotlib.pyplot as plt
import seaborn as sns #for plotting
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical 
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,  Activation
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate
from keras.layers.normalization import BatchNormalization
from keras import layers

dataset_path = 'fer2013/fer2013.csv'
image_size=(48,48)
 
def load_fer2013():
    data = pd.read_csv(dataset_path)
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'),image_size)
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)

    emotions = pd.get_dummies(data['emotion']).values
    return faces, emotions
 
def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x
 
faces, emotions = load_fer2013()
faces = preprocess_input(faces)

#Step 2: Review sample data
#Read sample image from 2D array

emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

sample_file = faces[0]
sample_file_emotion_label = np.argmax(emotions[0])

print("This is a sample image recovered from the dataset in ndarray")
print("The emotion of this photo is:")
print(emotion_dict[sample_file_emotion_label])

#

root_dir = os.path.abspath('.')
output_dir = os.path.join(root_dir, 'image_print')

sample_file_2D = np.reshape(sample_file, (48, 48))
plt.imshow(sample_file_2D, cmap = "gray")
plt.show()

#w, h = 48, 48
#image = np.zeros((h, w), dtype=np.uint8)
#id = 1


#img = Image.fromarray(np.uint8(sample_file)).convert('RGB')

#print(img)

x_train_and_validate, x_test, y_train_and_validate, y_test = train_test_split(faces, emotions, test_size=0.2, shuffle=True)

x_train, x_validate, y_train, y_validate = train_test_split(x_train_and_validate, y_train_and_validate, test_size=0.2, shuffle=True)

#Step 3: model building

# data generator
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

input_img = Input(shape = (48, 48, 1))
x = Conv2D(32, (5,5), activation='relu', padding='same', name='Conv1_1')(input_img)
x = Conv2D(32, (3,3), activation='relu', padding='same', name='Conv1_2')(x)
x = MaxPooling2D((2,2), name='pool1')(x)

x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv2_1')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv2_2')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2), name='pool2')(x)

x = Conv2D(128, (3,3), activation='relu', padding='same', name='Conv3_1')(x)
x = BatchNormalization(name='bn1')(x)
x = Conv2D(128, (3,3), activation='relu', padding='same', name='Conv3_2')(x)
x = BatchNormalization(name='bn2')(x)
x = MaxPooling2D((2,2), name='pool3')(x)
   
     
x = Flatten(name='flatten')(x)
x = Dense(1024, activation='relu', name='fc1')(x)
x = Dropout(0.6, name='dropout1')(x)
x = Dense(256, activation='relu', name='fc2')(x)
x = Dropout(0.4, name='dropout2')(x)
x = Dense(7, activation='softmax', name='fc3')(x)

model =  Model(inputs=input_img, outputs=x)
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
print(model.summary())

epochs = 20
batch_size = 32

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

model.fit_generator(data_generator.flow(x_train, y_train, batch_size),
                        steps_per_epoch = len(x_train) / batch_size,
                        epochs = epochs, verbose=1, callbacks=[learning_rate_reduction],
                        validation_data=(x_validate, y_validate))


model.save_weights('keras_model_emotion_weights_v2.h5')



model.load_weights('keras_model_emotion_weights_v2.h5')

raw_prediction = model.predict(x_test)
prediction_class = np.argmax(raw_prediction, axis = 1)

y_test = np.argmax(y_test, axis = 1)

output_df = pd.DataFrame(np.column_stack([y_test, prediction_class]), columns = ["Actual_class", "Label"])
output_df.loc[output_df['Actual_class'] == output_df['Label'], 'Correctness'] = 1
output_df.loc[output_df['Actual_class'] != output_df['Label'], 'Correctness'] = 0

accuracy = output_df["Correctness"].sum() / output_df.shape[0]
             
print("The accuracy of classification:")
print(accuracy)
             
output_df.to_csv("Output.csv", index = False)