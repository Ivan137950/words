import array
import os
from re import L


imdb_dir="/Users/limon/OneDrive/Рабочий стол/GitHubProg/words/aclImdb"
train_dir="/Users/limon/OneDrive/Рабочий стол/GitHubProg/words/aclImdb/test"

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            with open(os.path.join(dir_name, fname), 'r', encoding='utf-8') as f:
                texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen = 100 # amount of read words
training_samples = 200  
validation_samples = 10000 
max_words = 10000 # 10000 most popular words 

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index  # the array of sequences

print('Found %s unique tokens.' % len(word_index))  #72 633 unique squences

data = pad_sequences(sequences, maxlen=maxlen)
# print(data)
labels = np.asarray(labels)

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape) 

indices =np.arange(data.shape[0]) # [    0     1     2 ... 17240 17241 17242]
np.random.shuffle(indices)
data = data[indices]   #  mixed data and labels
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

# already done dataset
embeddings_index={}
glove_dir="/Users/limon/Downloads/glove.6B"
with open(os.path.join(glove_dir, 'glove.6B.100d.txt'), 'r', uncoding='utf-8') as f :
    for line in f:
        words = line.split()
        word = words[0]
        coefs = np.array(words[1:],'float32')
        embeddings_index[word] = coefs
f.close()
print(embeddings_index)
print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False