from convokit import Corpus, download
import numpy as np
import tensorflow as tf
import re

PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
WINDOW_SIZE = 30

def preprocess_sentence(sentence):
    """
    Takes out any contractions and separates punctuation from the word (separates on whitespace)
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
    sentence = re.sub(r"what's", "that is", sentence)
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
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    return sentence


def pad_corpus(sentence, count):
    """
    param: sentence split on whitespace
    """
    padded_sentence = sentence[:WINDOW_SIZE-1]
    if (count%2 == 0):
        padded_sentence += [STOP_TOKEN] + [PAD_TOKEN] * (WINDOW_SIZE - len(padded_sentence)-1)
    else:
        padded_sentence = [START_TOKEN] + padded_sentence + [STOP_TOKEN] + [PAD_TOKEN] * (WINDOW_SIZE - len(padded_sentence)-1)
    return padded_sentence


def build_vocab(sentences):
    """
  	Builds vocab from list of sentences

	:param sentences:  list of sentences, each a list of words
	:return: tuple of (dictionary: word --> unique index, pad_token_idx)
  	"""
    tokens = []
    for s in sentences: tokens.extend(s)
    all_words = sorted(list(set([STOP_TOKEN,PAD_TOKEN,UNK_TOKEN] + tokens)))

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


def get_data():
    """
    :return: Tuple of train containing:
	(2-d list or array with training input sentences in vectorized/id form [num_sentences x 30] ),
	(2-d list or array with training label sentences in vectorized/id form [num_sentences x 31]),
	(2-d list or array with test input sentences in vectorized/id form [num_sentences x 30]),
	(2-d list or array with test label sentences in vectorized/id form [num_sentences x 31]),
	vocab (Dict containg word->index mapping),
	padding ID (the ID used for *PAD* in the vocab. This will be used for masking loss)
	"""
    corpus = Corpus(filename=download("friends-corpus"))
    count = 0
    inputs = []
    labels = []
    for utt in corpus.iter_utterances():
        utt_preprocessed = preprocess_sentence(utt.text)
        utt_split = utt_preprocessed.split()
        utt_pad = pad_corpus(utt_split, count)
        if (count%2 == 0):
            inputs.append(utt_pad)
        else: 
            labels.append(utt_pad)
        count += 1
    if (len(inputs) != len(labels)):
        m = min(len(inputs), len(labels))
        inputs = inputs[:m]
        labels = labels[:m]
    # this would make it so there are no unk_vocabs 
    vocab, pad_indx = build_vocab(inputs+labels)
    split_indx = int(len(inputs) * .9)
    train_inputs = inputs[:split_indx]
    test_inputs = inputs[split_indx:]
    train_labels = labels[:split_indx]
    test_labels = labels[split_indx:]
    train_inputs = convert_to_id(vocab, train_inputs)
    test_inputs = convert_to_id(vocab, test_inputs)
    train_labels = convert_to_id(vocab, train_labels)
    test_labels = convert_to_id(vocab, test_labels)
    
    return train_inputs, test_inputs, train_labels, test_labels, vocab, pad_indx

