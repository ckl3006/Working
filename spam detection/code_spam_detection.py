import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import string


from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#Step 1: input
input_df = pd.read_csv('spam.csv',delimiter=',',encoding='latin-1')
#print(input_df.head())

print(input_df.columns)

input_df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
print(input_df.info())
#print(input_df.head())

#Step 2: EDA
sns.countplot(input_df.v1)
plt.title('Number of ham and spam messages')

#Step 3: Train-Test Split
input_x = input_df['v2']
input_y = input_df['v1']

#remove punctutations
input_x = input_x.str.replace('[^\w\s]','')
#change upper case to lower case
input_x = input_x.str.lower()

#print(input_x.iloc[0])

encoder = LabelEncoder()
input_y = encoder.fit_transform(input_y)
#print(input_y.shape)

max_words = 1500
max_len = 150

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(input_x)

x_train, x_test, y_train, y_test = train_test_split(input_x, input_y, test_size=0.15)
input_y = input_y.reshape(-1, 1)
print(input_y.shape)

#Step 4: Word Embedding

x_train = tokenizer.texts_to_sequences(x_train)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)

#print(type(x_train))
print(x_train)

#Step 5: Model Building
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    f = open(path, encoding= 'utf8')
    
    return dict(get_coefs(*line.strip().split(' ')) for line in f)

def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            pass
    return embedding_matrix

def build_model(embedding_matrix):
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], trainable=False)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.4)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

#root directory for storing the embedding file, to be updated
root_dir_for_embedding = "D:\\toxicity prediction"
embedding_file = "glove.840B.300d.txt"

embedding_path = os.path.join(root_dir_for_embedding, embedding_file)

embedding_matrix = build_matrix(tokenizer.word_index, embedding_path)
#Note: the embedding matrix got the shape as (9562, 200)

model = build_model(embedding_matrix)
model.summary()
print(model.layers[1])
model.layers[1].set_weights([embedding_matrix])
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=128,epochs=10,
          validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

model.save('model_v1.h5')

#Final: Check the accuracy
x_test = tokenizer.texts_to_sequences(x_test)
x_test = sequence.pad_sequences(x_test,maxlen=max_len)

accuracy = model.evaluate(x_test,y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accuracy[0],accuracy[1]))
