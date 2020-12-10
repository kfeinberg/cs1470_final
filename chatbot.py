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

    if len(sys.argv) > 1:
        mode = 'TF'
    else:
        mode = 'no TF'

    train_inputs, test_inputs, train_labels, test_labels, vocab, pad_indx = get_data(mode = 'MT')
    model = Transformer_Model(WINDOW_SIZE, len(vocab), 1)
    model.load_weights('saved/my_model')

    # dictionary of val -> word
    lookup = dict([(value, key) for key, value in vocab.items()])

    try:
        while True:
            val = input("user: ")

            # process user input to match model input
            val = preprocess_sentence(val)
            val = val.split()
            val = pad_corpus_chatbot(val, 0) # input padding
            val = convert_to_id_single(vocab, val)

            val = np.reshape(val, (1, len(val)))

            # generate model response to user input
            res = generate_sentence(model, val, lookup, vocab, mode)
            res = convert_to_words(res, lookup)
            print('model: ' + res)

    except KeyboardInterrupt:
        print('\ngoodbye!')

# given encoder input and lookup dictionary, returns model-generated numeric sentence
def generate_sentence(model, encoder_input, lookup, vocab, mode):

    # decoder input starts as start token + padding
    decoder_input = [START_TOKEN] + [PAD_TOKEN] * (80-1)
    decoder_input = convert_to_id_single(vocab, decoder_input)
    decoder_input = np.reshape(decoder_input, (1, len(decoder_input)))

    for i in range(1, 80-1):
        if (mode == 'TF'):
            res = np.array(model.call(encoder_input, encoder_input, mode = 'MT', is_training=False))
            res = np.argmax(res, axis=2)[0]
            return res
        else:
            res = np.array(model.call(encoder_input, decoder_input, mode = 'MT', is_training=False)) # teacher forcing removed
            res = np.argmax(res, axis=2)[0]
            decoder_input[0][i] = res[i-1] # sets ith index of decoder
            converted_symbol = lookup[decoder_input[0][i]]
            if (converted_symbol == STOP_TOKEN): # reached end of sentence
                return res

    return res

# converts model encoded sentence to English sentence
def convert_to_words(sentence, lookup):
    res = ''
    for val in sentence:
        converted = lookup[val]
        if (converted == STOP_TOKEN):
            return res
        res = res + converted + ' '
    return res

if __name__ == '__main__':
	main()
