import os
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import gensim

import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.utils import shuffle

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras import regularizers


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#Step 1: input
root_dir = os.path.abspath('.')
data_dir = os.path.join(root_dir, 'data')
dataset_path = os.path.join(data_dir, 'training.1600000.processed.noemoticon.csv')

input_df = pd.read_csv(dataset_path, encoding = "ISO-8859-1", names = ["target", "ids", "date", "flag", "user", "text"])

print(input_df.head())
#target: 0 means negative, 4 means positive
print(input_df['target'].unique())
print(input_df['flag'].unique())

# testing only: only read a few row for memory and time saving
input_df = shuffle(input_df)
input_df = input_df[0:100000]

#Step 2: EDA
decode_target = {0: "NEGATIVE", 4: "POSITIVE"}
def decode_sentiment(label):
    return decode_target[int(label)]

input_df_temp = input_df['target'].apply(lambda x: decode_sentiment(x))

target_cnt = Counter(input_df_temp)

#Note: positive / negative counts are 800000
plt.figure(figsize=(16,8))
plt.bar(target_cnt.keys(), target_cnt.values())
plt.title("Dataset labels distribuition")

del input_df_temp

#Step 3: data cleaning

hyperlink_cleaning = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")

def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(hyperlink_cleaning, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

def clean_special_chars(text):
    for p in punct:
        text = text.replace(p, ' ')
    return text

def lower_conversion(text):
    return text.str.lower()

input_df['text'] = input_df['text'].apply(lambda x: preprocess(x))
input_df['text'] = input_df['text'].apply(lambda x: clean_special_chars(x))
#input_df['text'] = input_df['text'].str.lower()
print(input_df['text'].head())
print(input_df['text'].iloc[0])

input_df = input_df.dropna()

X = input_df.iloc[:,[5]]
Y = input_df.iloc[:,0]
Y[Y == 4] = 1 #adjust class positive to 1 directly
print(Y.head())



#Step 4: Word Embedding / Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
print("TRAIN size:", len(X_train))
print("TEST size:", len(X_test))
print(type(X_train))
print(type(X_test))

documents = [_text.split() for _text in X_train.text] 
w2v_model = gensim.models.word2vec.Word2Vec(size = 300, 
                                            window = 7, 
                                            min_count = 10, 
                                            workers = 8)
w2v_model.build_vocab(documents)

w2v_model.train(documents, total_examples = len(documents), epochs = 32)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train.text)

vocab_size = len(tokenizer.word_index)
print("Total words", vocab_size)

max_length = 300
X_train = tokenizer.texts_to_sequences(X_train.text)
X_train = pad_sequences(X_train, maxlen=max_length)

#x_test = pad_sequences(tokenizer.texts_to_sequences(df_test.text), maxlen=max_length)

#encoder = LabelEncoder()
#encoder.fit(df_train.target.tolist())


#Step 5: model building

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    f = open(path, encoding= 'utf8')
    
    return dict(get_coefs(*line.strip().split(' ')) for line in f)

def build_matrix(word_index, embedding_model):

    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_model[word]
        except KeyError:
            pass
    return embedding_matrix

def build_model(embedding_matrix):
    model = Sequential()
    model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], input_length = embedding_matrix.shape[1], dropout=0.2))
    model.add(Dropout(0.3))
    model.add(Conv1D(300,3,activation='relu'))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())
    model.add(Dense(200, activation='sigmoid'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])
    return model

embed_dim = 300

#embedding_matrix = build_matrix(tokenizer.vocabulary_, embedding_path)
embedding_matrix = build_matrix(tokenizer.word_index, w2v_model)
print(embedding_matrix.shape)

print("Finish embedding matrix")

model = build_model(embedding_matrix)

print(model.summary())




batch_size = 100

callbacks_parameters = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
              EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]

model.fit(X_train, Y_train, epochs = 20, batch_size=batch_size, verbose = 1, validation_split=0.2, callbacks=callbacks_parameters)

#Prediction
model.save('model_LSTM_v1.h5')

#Final: Check the accuracy

X_test = tokenizer.texts_to_sequences(X_test.text)
X_test = pad_sequences(X_test, maxlen=max_length)

accuracy = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accuracy[0],accuracy[1]))