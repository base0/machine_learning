'''
credit : [@nagarindkx](https://github.com/nagarindkx)
ู[โค้ดเดิม](https://github.com/nagarindkx/python/blob/master/python%2006%20Sentiment%20Analysis.ipyn
[บทความ](https://sysadmin.psu.ac.th/2019/01/15/python-06-sentiment-analysis-with-keras-tensorflow/)
'''

import keras
from keras import *
from keras.layers import *

model = keras.models.load_model('rnn.h5')
model.summary()

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import numpy as np

negative = [
    ['I do not like it', 0],
    ['bad movie', 0],
    ['I hate it', 0],
    ['Not good at all', 0]
]
neutral = [
    ['not bad', 1],
    ['so so', 1],
    ['OK', 1],
    ['no comment', 1]
]
positive = [
    ['Good movie', 2],
    ['I love it', 2],
    ['Like', 2],
    ['Two thumbs up', 2]
]
data = np.array(negative + neutral + positive)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data[:,0])
tokenizer.word_index
maxlen = max([len(s) for s in tokenizer.texts_to_sequences(data[:,0])])

def sentiment(t):
    x_test=[]
    s = []                               # t :                        'I do not like it'
    for w in text_to_word_sequence(t):   # text_to_word_sequence(t) : ['I', 'do', 'not', 'like', 'it']
        s.append(tokenizer.word_index[w] if w in tokenizer.word_index else 0)
    x_test.append(s)
    x_test = pad_sequences(x_test,maxlen=maxlen, padding='post')
    return model.predict_classes(x_test)[0]


# A very simple Flask Hello World app for you to get started with...

from flask import Flask
from flask import request


app = Flask(__name__)

@app.route('/')
def hello_world():
    s = request.args.get('s')
    return str(sentiment(s))


