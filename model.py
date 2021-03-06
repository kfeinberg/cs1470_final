from convokit import Corpus, download
import numpy as np
import tensorflow as tf
import transformer_funcs as transformer

from attenvis import AttentionVis
av = AttentionVis()

class Transformer_Model(tf.keras.Model):
    def __init__(self, window_size, vocab_size, len_data):
        super(Transformer_Model, self).__init__()

        # train and test sentences will have same vocab_size and window_size
        self.vocab_size = vocab_size
        self.window_size = window_size # 30

        # hidden layer size, batch_size, embedding_size, optimizer
        self.hidden_size = 128
        self.batch_size = 102
        self.embedding_size = 501

        # piecewise constant decay learning rate
        num_steps = len_data / self.batch_size
        boundaries = [80*num_steps, 120*num_steps, 160*num_steps, 180*num_steps]
        values = [1e-3, 1e-4, 8e-5, 1e-5, 5e-6]
        self.lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries, values)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)

        # prompt (input) and response (label) embedding layers:
        self.prompt_embedding = tf.Variable(tf.random.normal([self.vocab_size,self.embedding_size], stddev=.01, dtype=tf.float32))
        self.response_embedding = tf.Variable(tf.random.normal([self.vocab_size,self.embedding_size], stddev=.01, dtype=tf.float32))

        # Positional Encoding layers
        self.prompt_pos_enc = transformer.Position_Encoding_Layer(self.window_size, self.embedding_size)
        self.response_pos_enc = transformer.Position_Encoding_Layer(self.window_size, self.embedding_size)

        # Encoder and Decoder layers
        self.encoder = transformer.Transformer_Block(self.embedding_size, is_decoder=False, multi_headed=True)
        self.decoder = transformer.Transformer_Block(self.embedding_size, is_decoder=True, multi_headed=True)

        # Dense layers
        self.dense1 = tf.keras.layers.Dense(units=self.hidden_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=self.vocab_size, activation='softmax')

        # LM encoder/decoder
        self.lstm = tf.keras.layers.LSTM(self.hidden_size, return_sequences=True, return_state=True)

    @tf.function
    def call(self, encoder_input, decoder_input, mode, is_training=True):
        """
		:param encoder_input: batched ids corresponding to prompt sentences
		:param decoder_input: batched ids corresponding to response sentences
		:return probs: The 3d probabilities as a tensor, [batch_size x window_size x vocab_size]
		"""
        # if MT, then use encoder
        if mode == 'MT':
            # 1) embed prompt sentences and add positional encoding
            prompt_embedded = tf.nn.embedding_lookup(self.prompt_embedding, encoder_input)
            pos_enc_prompts = self.prompt_pos_enc.call(prompt_embedded)

            # 2) pass prompt embeddings to encoder
            encoder_output = self.encoder.call(pos_enc_prompts, is_training=is_training)

        # decoder
        # 3) embed response sentences and add positional encoding
        response_embedded = tf.nn.embedding_lookup(self.response_embedding, decoder_input)
        pos_enc_responses = self.response_pos_enc.call(response_embedded)

        if mode == 'MT':
            # 4) pass response embeddings and encoder output to the decoder
            decoder_output = self.decoder.call(pos_enc_responses, context=encoder_output, mode=mode, is_training=is_training)
        elif mode == 'LM':
            # no context
            decoder_output, final_memory_state, final_carry_state = self.lstm(response_embedded, initial_state=None)
            #decoder_output = self.decoder.call(pos_enc_responses, context=None, mode=mode)

        # 5) pass through dense layers to get probabilities
        output = self.dense1(decoder_output)
        probs = self.dense2(output)

        return probs

    def accuracy_function(self, probs, responses, mask):
        """
        Calculate accuracy of one batch
        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x vocab_size]
		:param responses:  integer tensor, word prediction responses [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: scalar tensor of accuracy of the batch between 0 and 1
        """

        predictions = tf.argmax(input=probs, axis=2) # axis=2 bc along vocab_size dim
        accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(predictions, responses), dtype=tf.float32),mask))
        return accuracy

    def loss_function(self, probs, responses, mask):

        return tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(responses, probs) * mask)

    @av.call_func
    def __call__(self, *args, **kwargs):
        return super(Transformer_Model, self).__call__(*args, **kwargs)
