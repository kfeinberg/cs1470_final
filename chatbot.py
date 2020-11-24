import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from model import Transformer_Model
import sys
import random
from keras.models import load_model
from preprocess import preprocess_sentence, convert_to_id
from main import test

def main():

    train_inputs, test_inputs, train_labels, test_labels, vocab, pad_indx = get_data()

    model = Transformer_Model(WINDOW_SIZE, len(vocab))
    test(model, test_inputs, test_labels, pad_indx)
    model.load_weights('saved/my_model')

    # dictionary of val -> word
    lookup = dict([(value, key) for key, value in vocab.items()])

    try:
        while True:
            val = input("user input to model: ")
            val = preprocess_sentence(val)
            val = val.split()
            val = pad_corpus(val, 0)
            val = convert_to_id_single(vocab, val)
            val = np.reshape(val, (1, len(val)))
            res = model.call(val, val) # todo: call function without decoder input
            res = np.argmax(res, axis=2)
            res = convert_to_words(res, lookup)
            print(res)

    except KeyboardInterrupt:
        print('\ngoodbye!')

def convert_to_words(sentence, lookup):
    res = ''
    for val in sentence[0]:
        res = res + lookup[val] + ' '
    return res

if __name__ == '__main__':
	main()
