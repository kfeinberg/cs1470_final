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

def train(model, train_inputs, train_labels, padding_index, mode):
    num_sentences = train_inputs.shape[0]
    len_label_sentences = train_labels.shape[1]

    loss_list = []
    accuracy_list = []
    total_words = 0

    # shuffle inputs and labels
    indices = tf.range(num_sentences)
    indices = tf.random.shuffle(indices)
    shuff_ins = tf.gather(train_inputs, indices)
    shuff_labs = tf.gather(train_labels, indices)

    for batch in range(0, num_sentences - model.batch_size, model.batch_size):
        start = batch
        end = batch + model.batch_size

        if mode == 'MT':
            encoder_input = shuff_ins[start:end]
            decoder_input = shuff_labs[start:end, 0:len_label_sentences - 1]
            decoder_labels = shuff_labs[start:end, 1:]
        elif mode == 'LM':
            encoder_input = None
            decoder_input = shuff_ins[start:end]
            decoder_labels = shuff_labs[start:end]

        with tf.GradientTape() as tape:
            # forward pass
            probs = model.call(encoder_input, decoder_input, mode)

            # only mask MT bc no padding for LM
            if mode == 'MT':
                mask = decoder_labels != padding_index
            elif mode == 'LM':
                mask = tf.cast(decoder_labels, dtype=tf.float32)

            num_words = np.sum(mask)
            total_words += num_words
            mask = tf.cast(mask, dtype=tf.float32)
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
def test(model, test_inputs, test_labels, padding_index, mode):
    num_sentences = test_inputs.shape[0]
    len_label_sent = test_labels.shape[1]

    loss_list = []
    accuracy_list = []
    total_words = 0

    for batch in range(0, num_sentences - model.batch_size, model.batch_size):
        start = batch
        end = batch + model.batch_size

        if mode == 'MT':
            encoder_input = test_inputs[start:end]
            decoder_input = test_labels[start:end, 0:len_label_sent - 1]
            decoder_labels = test_labels[start:end, 1:]
        elif mode == 'LM':
            encoder_input = None
            decoder_input = test_inputs[start:end]
            decoder_labels = test_labels[start:end]

        probs = model.call(encoder_input, decoder_input, mode)

        # only mask MT bc no padding for LM
        if mode == 'MT':
            mask = decoder_labels != padding_index
        elif mode == 'LM':
            mask = tf.cast(decoder_labels, dtype=tf.float32)

        num_words = np.sum(mask)
        total_words += num_words

        mask = tf.cast(mask, dtype=tf.float32)
        loss = model.loss_function(probs, decoder_labels, mask)
        loss_list.append(loss)

        batch_accuracy = model.accuracy_function(probs, decoder_labels, mask)
        batch_correct = batch_accuracy * num_words
        accuracy_list.append(batch_correct)

    perplexity = np.exp(np.sum(loss_list) / total_words)
    accuracy = np.sum(accuracy_list) / total_words

    return perplexity, accuracy

def main():

    # either LM (language modelling - only decoder) or MT (machine translation - encoder + decoder)
    mode = sys.argv[1]

    print("Running preprocessing...")
    train_inputs, test_inputs, train_labels, test_labels, vocab, pad_indx = get_data(mode)
    print("Preprocessing complete.")

    if mode == 'LM':
        train_inputs = np.asarray(train_inputs)
        train_labels = np.asarray(train_labels)
        test_inputs = np.asarray(test_inputs)
        test_labels = np.asarray(test_labels)

    print(test_inputs.shape)
    model = Transformer_Model(WINDOW_SIZE, len(vocab), train_inputs.shape[0])

    for epoch in range(1):
        print('Training epoch #' + str(epoch + 1))
        perp, acc = train(model, train_inputs, train_labels, pad_indx, mode)
        print(perp)
        print(acc)

    print('Testing now')
    perplexity, accuracy = test(model, test_inputs, test_labels, pad_indx, mode)
    print(perplexity)
    print(accuracy)

    # use SAVE as command line argument to save model
    if len(sys.argv) == 3 and sys.argv[2] == 'SAVE':
        model.save_weights('saved/my_model')

if __name__ == '__main__':
	main()
