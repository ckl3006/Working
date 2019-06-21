import os
import glob
from pickle import load, dump
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import preprocess_input
import string
from keras.preprocessing import image
from numpy import array
from time import time
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

def load_descriptions(doc):
	mapping = dict()
	# process lines
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		if len(line) < 2:
			continue
		# take the first token as the image id, the rest as the description
		image_id, image_desc = tokens[0], tokens[1:]
		# extract filename from image id
		image_id = image_id.split('.')[0]
		# convert description tokens back to string
		image_desc = ' '.join(image_desc)
		# create the list if needed
		if image_id not in mapping:
			mapping[image_id] = list()
		# store description
		mapping[image_id].append(image_desc)
	return mapping

def clean_descriptions(descriptions):
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for key, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]
			# tokenize
			desc = desc.split()
			# convert to lower case
			desc = [word.lower() for word in desc]
			# remove punctuation from each token
			desc = [w.translate(table) for w in desc]
			# remove hanging 's' and 'a'
			desc = [word for word in desc if len(word)>1]
			# remove tokens with numbers in them
			desc = [word for word in desc if word.isalpha()]
			# store as string
			desc_list[i] =  ' '.join(desc)

def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()



# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions

def preprocess(image_path):
    # Convert all the images to size 224x224x3 as expected by the inception v3 model
    img = image.load_img(image_path, target_size=(224, 224, 3))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x

def encode(image):
    image = preprocess(image) # preprocess the image
    fea_vec = model.predict(image) # Get the encoding vector for the image
    #print(fea_vec.shape)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    #print(fea_vec.shape)
    return fea_vec

def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# calculate the length of the description with the most words
def maximum_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

def load_photo_features(filename, dataset):
	# load all features
    all_features = load(open(filename, 'rb'))
	# filter features
    features = {k: all_features[k] for k in dataset if k in all_features}
    return features

def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n=0
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            n+=1
            # retrieve the photo feature
            photo = photos[key]
            #photo = photos[key+'.jpg']
            for desc in desc_list:
                # encode the sequence
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            # yield the batch data
            if n==num_photos_per_batch:
                yield [[array(X1), array(X2)], array(y)]
                #print("The shape of array X1, X2 and y")
                print(array(X1).shape)
                print(array(X2).shape)
                print(array(y).shape)
                X1, X2, y = list(), list(), list()
                n=0

def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = final_model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

root_dir = os.path.abspath('.')
train = set()
text_dir = os.path.join(root_dir, 'Flickr8k_text')
filename = os.path.join(text_dir, 'Flickr8k.token.txt')
# load descriptions
doc = load_doc(filename)
descriptions = load_descriptions(doc)
clean_descriptions(descriptions)

save_descriptions(descriptions, 'descriptions.txt')

train_and_test_file = os.path.join(text_dir, 'Flickr_8k.trainImages.txt')
train_and_test = load_set(filename)
train = set()
test = set()
count = 0
image_dir = os.path.join(root_dir, 'Flicker8k_Dataset')
for record in train_and_test:
	filename = os.path.join(image_dir, record)
	filename = filename + '.jpg'
	if count < 8000:
		a = len(train)
		if os.path.isfile(filename) == True:
			train.add(record)
			if len(train) - a > 0:
				count += 1
	else:
		b = len(test)
		if os.path.isfile(filename) == True:
			test.add(record)
			if len(test) - b > 0:
				count += 1

train_descriptions = load_clean_descriptions('descriptions.txt', train)
all_train_captions = []
for key, val in train_descriptions.items():
	for cap in val:
		all_train_captions.append(cap)
len(all_train_captions)

word_count_threshold = 5
word_counts = {}
nsents = 0
for sent in all_train_captions:
	nsents += 1
	for w in sent.split(' '):
		word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

ixtoword = {}
wordtoix = {}

ix = 1
for w in vocab:
	wordtoix[w] = ix
	ixtoword[ix] = w
	ix += 1

vocab_size = len(ixtoword) + 1 # one for appended 0's
max_length = maximum_length(train_descriptions)
max_length = 34

embeddings_index = {} # empty dictionary
f = open(os.path.join(root_dir, 'glove.6B.200d.txt'), encoding="utf-8")
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()

embedding_dim = 200

# Get 200-dim dense vector for each of the 10000 words in out vocabulary
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in wordtoix.items():
	#if i < max_words:
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		# Words not found in the embedding index will be all zeros
		embedding_matrix[i] = embedding_vector
        

test = set()
test.add("test")
#list of train / test in full path variables
image_dir = os.path.join(root_dir, 'test_img')


                
test_img = []

for record in test:
	record = os.path.join(image_dir, record)
	record = record + '.jpg'
	test_img.append(record)



model = DenseNet121()
model.layers.pop()
model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

start = time()
encoding_test = {}
encoding_test_count = 0
for img in test:
	full_img_path = os.path.join(image_dir, img)
	full_img_path = full_img_path + '.jpg'
	encoding_test[img] = encode(full_img_path)
	encoding_test_count += 1


encoded_test_images_dir = os.path.join(root_dir, "encoded_test_images.pkl")

with open(encoded_test_images_dir, "wb") as encoded_pickle:
    dump(encoding_test, encoded_pickle)



test_features_dir = os.path.join(root_dir, "encoded_test_images.pkl")

test_features = load(open(test_features_dir, "rb"))

inputs1 = Input(shape=(1024,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
final_model = Model(inputs=[inputs1, inputs2], outputs=outputs)

final_model.summary()

final_model.layers[2].set_weights([embedding_matrix])
final_model.layers[2].trainable = False

final_model.compile(loss='categorical_crossentropy', optimizer='adam')

final_model = load_model('model_3.h5')

#Test the image with filename "test.jpg" in the folder "test_img"
z = 0
pic = list(test_features.keys())[z]
test_image = test_features[pic].reshape((1,1024))
#print(test_image.shape)
full_test_image_path = os.path.join(image_dir, pic)
full_test_image_path = full_test_image_path + '.jpg'
x=plt.imread(full_test_image_path)

plt.imshow(x)
plt.show()

print("Image Caption:",greedySearch(test_image))