import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.callbacks import LambdaCallback
import random, sys

with open("male.txt", "r") as file:
    names_text = file.read()
chars = sorted(list(set(names_text)))
char_indices = {}
indices_char = {}
for i, char in enumerate(chars):
    char_indices[char] = i
    indices_char[i] = char
maxlen = 10
step = 3
sentences = []
next_chars = []

for i in range(0, len(names_text) - maxlen, step):
    sentences.append(names_text[i: i + maxlen])
    next_chars.append(names_text[i + maxlen])

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=bool)
y = np.zeros((len(sentences), len(chars)), dtype=bool)
for i,(sentence, next_char) in enumerate(zip(sentences, next_chars)):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_char]] = 1

model = Sequential([
    SimpleRNN(128, input_shape=(maxlen, len(chars))),
    Dense(len(chars), activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam')

def generate(epoch, _):
    start_index = random.randint(0, len(names_text) - maxlen - 1)
    sentence = names_text[start_index: start_index + maxlen]
    print('Generating with seed:', sentence)
    for i in range(100):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.
        preds = model.predict(x_pred, verbose=0)[0]
        preds = np.log(np.asarray(preds).astype('float64')) / 0.5
        next_char = indices_char[np.argmax(np.random.multinomial(1, np.exp(preds) / np.sum(np.exp(preds)), 1))]
        sentence = sentence[1:] + next_char
        sys.stdout.write(next_char)
    print()

name_generator = LambdaCallback(on_epoch_end=generate)
model.fit(x, y, batch_size=128, epochs=20, callbacks=[name_generator])
