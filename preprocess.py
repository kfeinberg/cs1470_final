from convokit import Corpus, download
import numpy as np
import tensorflow as tf
import re
from collections import Counter

from attenvis import AttentionVis
av = AttentionVis()

PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
WINDOW_SIZE = 80 #15
HISTORY_SIZE = 5 # number of preceding sentences as history

def preprocess_sentence(sentence):
    """
    Takes out any contractions and separates punctuation from the word (separates on whitespace)
    :param sentence: the sentence to be separated/edited
    :return: sentence without contractions, and punctuation surrounded by whitespace.
    """
    sentence = sentence.lower().strip()
	# creating a space between a word and the punctuation following it
	# eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # removing contractions
    sentence = re.sub(r"i'm", "i am", sentence)
    sentence = re.sub(r"he's", "he is", sentence)
    sentence = re.sub(r"she's", "she is", sentence)
    sentence = re.sub(r"it's", "it is", sentence)
    sentence = re.sub(r"that's", "that is", sentence)
    sentence = re.sub(r"what's", "what is", sentence)
    sentence = re.sub(r"where's", "where is", sentence)
    sentence = re.sub(r"how's", "how is", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can't", "cannot", sentence)
    sentence = re.sub(r"n't", " not", sentence)
    sentence = re.sub(r"n'", "ng", sentence)
    sentence = re.sub(r"'bout", "about", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    # sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence) # allows ?.!,
    sentence = re.sub(r"[^a-zA-Z]+", " ", sentence) # doesn't allow ?.!,
    sentence = sentence.strip()
    return sentence


# def pad_corpus(sentence, count):
#     """
#     Pads a sentence passed in with STOP and PAD tokens, as well as a START token if it is a label.
#     :param sentence: sentence split on whitespace
#     :param count: the iteration number for seeing if label or input
#     :return: the sentences shortened/lengthened to WINDOW_SIZE length and padded with stop/start/pad tokens
#     """
#     padded_sentence = sentence[:WINDOW_SIZE-1]
#     if (count%2 == 0):
#         padded_sentence += [STOP_TOKEN] + [PAD_TOKEN] * (WINDOW_SIZE - len(padded_sentence)-1)
#     else:
#         padded_sentence = [START_TOKEN] + padded_sentence + [STOP_TOKEN] + [PAD_TOKEN] * (WINDOW_SIZE - len(padded_sentence)-1)
#     return padded_sentence

def pad_corpus(inputs, labels):
    """
    Pads a sentence passed in with STOP and PAD tokens, as well as a START token if it is a label.
    :param inputs: list of inputs, each line is a concatenation of 5 sentences
    :param labels: list of label sentences
    :return: the sentences shortened/lengthened to WINDOW_SIZE length and padded with stop/start/pad tokens
    """
    INPUTS_padded_sentences = []

    for line in inputs:
        padded_INPUT = line[-WINDOW_SIZE:]
        if len(padded_INPUT) < WINDOW_SIZE:
            padded_INPUT += [PAD_TOKEN] * (WINDOW_SIZE - len(padded_INPUT))
        
        INPUTS_padded_sentences.append(padded_INPUT)

    LABELS_padded_sentences = []

    for line in labels:
        padded_LABEL = line[-WINDOW_SIZE:]
        # replace START token with speaker for labels
        padded_LABEL = [START_TOKEN] + padded_LABEL + [PAD_TOKEN] * (WINDOW_SIZE - len(padded_LABEL))
        LABELS_padded_sentences.append(padded_LABEL)

    return INPUTS_padded_sentences, LABELS_padded_sentences


def build_vocab(sentences):
    """
  	Builds vocab from list of sentences
	:param sentences:  list of sentences, each a list of words
	:return: tuple of (dictionary: word --> unique index, pad_token_idx)
  	"""
    tokens = []
    for s in sentences: tokens.extend(s)
    counted = Counter(tokens)
    new_tokens = [word for word in tokens if counted[word] > 5]
    all_words = sorted(list(set([STOP_TOKEN,PAD_TOKEN,UNK_TOKEN] + new_tokens)))

    vocab =  {word:i for i,word in enumerate(all_words)}

    return vocab,vocab[PAD_TOKEN]


def convert_to_id(vocab, sentences):
    """
    Convert sentences to indexed
    :param vocab:  dictionary, word --> unique index
    :param sentences:  list of lists of words, each representing padded sentence
    :return: numpy array of integers, with each row representing the word indeces in the corresponding sentences
    """
    return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])

def convert_to_id_single(vocab, sentence):
    """
    Converts one sentence to be indexed
    :param vocab:  dictionary, word --> unique index
    :param sentence:  list of words, one padded sentence
    :return: numpy array of integers, with each row representing the word indeces in the corresponding sentences
    """
    return np.stack([vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence])

def get_data(mode):
    """
	Use the helper functions in this file to read and parse training and test data, then pad the corpus.
	Then vectorize your train and test data based on your vocabulary dictionaries.

	:return: Tuple of train containing:
	(2-d list or array with training input sentences in vectorized/id form [num_sentences x 31] ),
	(2-d list or array with test input sentences in vectorized/id form [num_sentences x 31]),
	(2-d list or array with training label sentences in vectorized/id form [num_sentences x 30]),
	(2-d list or array with test label sentences in vectorized/id form [num_sentences x 30]),
	vocab (Dict containg word->index mapping),
	the padding ID (the ID used for *PAD* in the English vocab. This will be used for masking loss)
	"""
    corpus = Corpus(filename=download("friends-corpus"))
    inputs = []
    labels = []
    entire_corpus = []
    for i, utt in enumerate(corpus.iter_utterances()):
        # added first name of speaker to beginning of sentence and STOP token to end
        utt_preprocessed = utt.speaker.id.split()[0] + ' ' + preprocess_sentence(utt.text) + ' ' + STOP_TOKEN
        utt_split = utt_preprocessed.split()
        entire_corpus.append(utt_split)

    if mode == 'MT':
        print('Mode is machine translation')
        # split into 5-sentence inputs and 1 sentence label
        for i in range(0, len(entire_corpus) - HISTORY_SIZE):
            inputs.append(sum(entire_corpus[i: i + HISTORY_SIZE], []))
            labels.append(entire_corpus[i + HISTORY_SIZE])
        
        # padding
        inputs, labels = pad_corpus(inputs, labels)
    elif mode == 'LM': 
        print('Mode is language modelling')
        flattened_corpus = [word for senten in entire_corpus for word in senten]
        inputs = flattened_corpus[:len(entire_corpus) - 1]
        labels = flattened_corpus[1:]

        # make into WINDOW_SIZE inputs and labels
        input_windows = []
        label_windows = []
        for window_start in range(0, len(inputs) - WINDOW_SIZE, WINDOW_SIZE):
            input_windows.append(inputs[window_start:window_start + WINDOW_SIZE])
            label_windows.append(labels[window_start:window_start + WINDOW_SIZE])
        inputs = input_windows
        labels = label_windows

    # build the vocab dict
    vocab, pad_indx = build_vocab(inputs+labels)

    # split into training and testing sets
    split_indx = int(len(inputs) * .85)
    train_inputs = inputs[:split_indx]
    test_inputs = inputs[split_indx:]
    train_labels = labels[:split_indx]
    test_labels = labels[split_indx:]

    # convert sentences to list of IDs
    train_inputs = convert_to_id(vocab, train_inputs)
    test_inputs = convert_to_id(vocab, test_inputs)
    train_labels = convert_to_id(vocab, train_labels)
    test_labels = convert_to_id(vocab, test_labels)

    return train_inputs, test_inputs, train_labels, test_labels, vocab, pad_indx

    # corpus = Corpus(filename=download("friends-corpus"))
    # count = 0
    # inputs = []
    # labels = []
    # for utt in corpus.iter_utterances():
    #     utt_preprocessed = preprocess_sentence(utt.text)
    #     utt_split = utt_preprocessed.split()
    #     utt_pad = pad_corpus(utt_split, count)
    #     if (count%2 == 0):
    #         inputs.append(utt_pad)
    #     else:
    #         labels.append(utt_pad)
    #     count += 1
    # if (len(inputs) != len(labels)):
    #     m = min(len(inputs), len(labels))
    #     inputs = inputs[:m]
    #     labels = labels[:m]
    # # this would make it so there are no unk_vocabs
    # vocab, pad_indx = build_vocab(inputs+labels)
    # split_indx = int(len(inputs) * .9)
    # train_inputs = inputs[:split_indx]
    # test_inputs = inputs[split_indx:]
    # train_labels = labels[:split_indx]
    # test_labels = labels[split_indx:]
    # train_inputs = convert_to_id(vocab, train_inputs)
    # test_inputs = convert_to_id(vocab, test_inputs)
    # train_labels = convert_to_id(vocab, train_labels)
    # test_labels = convert_to_id(vocab, test_labels)

    # return train_inputs, test_inputs, train_labels, test_labels, vocab, pad_indx
