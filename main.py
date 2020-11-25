import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from model import Transformer_Model
import sys
import random

from attenvis import AttentionVis
av = AttentionVis()

def train(model, train_inputs, train_labels, padding_index):
    num_sentences = train_inputs.shape[0]
    len_label_sentences = train_labels.shape[1]

    loss_list = []
    accuracy_list = []
    total_words = 0

    for batch in range(0, num_sentences - model.batch_size, model.batch_size):
        start = batch
        end = batch + model.batch_size
        encoder_input = train_inputs[start:end]

        decoder_input = train_labels[start:end, 0:len_label_sentences - 1]
        decoder_labels = train_labels[start:end, 1:]

        with tf.GradientTape() as tape:
            # forward pass
            probs = model.call(encoder_input, decoder_input)
            mask = decoder_labels != padding_index #[0 if index == eng_padding_index else 1 for index in decoder_labels]
            num_words = np.sum(mask)
            total_words += num_words

            loss = model.loss_function(probs, decoder_labels, mask)
            loss_list.append(loss)

            batch_accuracy = model.accuracy_function(probs, decoder_labels, mask)
            batch_correct = batch_accuracy * num_words
            accuracy_list.append(batch_correct)
            print(loss)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    perplexity = np.exp(np.sum(loss_list) / total_words)
    accuracy = np.sum(accuracy_list) / total_words

    return perplexity, accuracy

@av.test_func
def test(model, test_inputs, test_labels, padding_index):
    num_sentences = test_inputs.shape[0]
    len_label_sent = test_labels.shape[1]

    loss_list = []
    accuracy_list = []
    total_words = 0

    for batch in range(0, num_sentences - model.batch_size, model.batch_size):
        start = batch
        end = batch + model.batch_size
        encoder_input = test_inputs[start:end]
        
        decoder_input = test_labels[start:end, 0:len_label_sent - 1] # take out last input bc no label to predict next word
        decoder_labels = test_labels[start:end, 1:] # take out first label bc no -1 input word to predict this label

        probs = model.call(encoder_input, decoder_input)
        mask = decoder_labels != padding_index #[0 if index == eng_padding_index else 1 for index in decoder_labels] #[batch_size x window_size]
        num_words = np.sum(mask)
        total_words += num_words

        loss = model.loss_function(probs, decoder_labels, mask)
        loss_list.append(loss)

        batch_accuracy = model.accuracy_function(probs, decoder_labels, mask)
        batch_correct = batch_accuracy * num_words
        accuracy_list.append(batch_correct)

    perplexity = np.exp(np.sum(loss_list) / total_words)
    accuracy = np.sum(accuracy_list) / total_words

    return perplexity, accuracy

def main():

    print("Running preprocessing...")
    train_inputs, test_inputs, train_labels, test_labels, vocab, pad_indx = get_data()
    print("Preprocessing complete.")

    model = Transformer_Model(WINDOW_SIZE, len(vocab), train_inputs.shape[0])

    for epoch in range(2):
        print('Training epoch #' + str(epoch + 1))
        perp, acc = train(model, train_inputs, train_labels, pad_indx)
        print(perp)
        print(acc)

    print('Testing now')
    perplexity, accuracy = test(model, test_inputs, test_labels, pad_indx)
    print(perplexity)
    print(accuracy)

    # use SAVE as command line argument to save model
    if len(sys.argv) == 2 and sys.argv[1] == 'SAVE':
        model.save_weights('saved/my_model')

if __name__ == '__main__':
	main()