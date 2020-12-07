import numpy as np
import tensorflow as tf
import numpy as np
import math

from attenvis import AttentionVis
av = AttentionVis()

@av.att_mat_func
def Attention_Matrix(K, Q, use_mask=False):

	window_size_queries = Q.get_shape()[1] # window size of queries
	window_size_keys = K.get_shape()[1] # window size of keys
	mask = tf.convert_to_tensor(value=np.transpose(np.tril(np.ones((window_size_queries,window_size_keys))*np.NINF,-1),(1,0)),dtype=tf.float32)
	atten_mask = tf.tile(tf.reshape(mask,[-1,window_size_queries,window_size_keys]),[tf.shape(input=K)[0],1,1])

	score = tf.matmul(Q, K, transpose_b=True) / math.sqrt(np.shape(K)[2])

	if use_mask:
		score = score + atten_mask

	return tf.nn.softmax(score)

class Atten_Head(tf.keras.layers.Layer):
	def __init__(self, input_size, output_size, use_mask):
		super(Atten_Head, self).__init__()

		self.use_mask = use_mask

		self.K_weight = self.add_weight("k_weight",shape=[input_size, output_size])
		self.Q_weight = self.add_weight("q_weight",shape=[input_size, output_size])
		self.V_weight =	self.add_weight("q_weight",shape=[input_size, output_size])

	@tf.function
	def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):

		K = tf.tensordot(inputs_for_keys, self.K_weight, axes=[[2],[0]])
		V = tf.tensordot(inputs_for_values, self.V_weight, axes=[[2],[0]])
		Q = tf.tensordot(inputs_for_queries, self.Q_weight, axes=[[2],[0]])

		attn = Attention_Matrix(K, Q, self.use_mask)
		return tf.matmul(attn, V)


class Multi_Headed(tf.keras.layers.Layer):
	def __init__(self, emb_sz, use_mask):
		super(Multi_Headed, self).__init__()

		self.embedding_sz = emb_sz
		self.use_mask = use_mask

		self.head1 = Atten_Head(self.embedding_sz, int(self.embedding_sz / 3), use_mask)
		self.head2 = Atten_Head(self.embedding_sz, int(self.embedding_sz / 3), use_mask)
		self.head3 = Atten_Head(self.embedding_sz, int(self.embedding_sz / 3), use_mask)

		self.dense = tf.keras.layers.Dense(emb_sz)

	@tf.function
	def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
		"""
		FOR CS2470 STUDENTS:
		This functions runs a multiheaded attention layer.
		Requirements:
			- Splits data for 3 different heads of size embed_sz/3
			- Create three different attention heads
			- Concatenate the outputs of these heads together
			- Apply a linear layer
		:param inputs_for_keys: tensor of [batch_size x WINDOW_SIZE x input_size ]
		:param inputs_for_values: tensor of [batch_size x WINDOW_SIZE x input_size ]
		:param inputs_for_queries: tensor of [batch_size x WINDOW_SIZE x input_size ]
		:return: tensor of [BATCH_SIZE x WINDOW_SIZE x output_size ]
		"""
		res1 = self.head1.call(inputs_for_keys, inputs_for_values, inputs_for_queries)
		res2 = self.head2.call(inputs_for_keys, inputs_for_values, inputs_for_queries)
		res3 = self.head3.call(inputs_for_keys, inputs_for_values, inputs_for_queries)

		full_res = tf.concat((res1, res2, res3), axis=2)

		return self.dense(full_res)


class Feed_Forwards(tf.keras.layers.Layer):
	def __init__(self, emb_sz):
		super(Feed_Forwards, self).__init__()

		self.layer_1 = tf.keras.layers.Dense(emb_sz,activation='relu')
		self.layer_2 = tf.keras.layers.Dense(emb_sz)

	@tf.function
	def call(self, inputs):
		layer_1_out = self.layer_1(inputs)
		layer_2_out = self.layer_2(layer_1_out)
		return layer_2_out

class Transformer_Block(tf.keras.layers.Layer):
	def __init__(self, emb_sz, is_decoder, multi_headed=False):
		super(Transformer_Block, self).__init__()

		self.ff_layer = Feed_Forwards(emb_sz)
		self.self_atten = Atten_Head(emb_sz,emb_sz,use_mask=is_decoder) if not multi_headed else Multi_Headed(emb_sz,use_mask=is_decoder)
		self.is_decoder = is_decoder
		if self.is_decoder:
			self.self_context_atten = Atten_Head(emb_sz,emb_sz,use_mask=False) if not multi_headed else Multi_Headed(emb_sz,use_mask=False)

		self.dropout = tf.keras.layers.Dropout(rate=0.1)
		self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)

	@tf.function
	def call(self, inputs, context=None, mode=None):
		"""
		If the multi_headed==True, the model uses multiheaded attention (Only 2470 students must implement this)
		:param inputs: tensor of [BATCH_SIZE x WINDOW_SIZE x EMBEDDING_SIZE ]
		:context: tensor of [BATCH_SIZE x WINDOW_SIZE x EMBEDDING_SIZE ] or None
			default=None, This is context from the encoder to be used as Keys and Values in self-attention function
		"""

		with av.trans_block(self.is_decoder):
			atten_out = self.self_atten(inputs,inputs,inputs)

		if self.is_decoder == False: # only dropout if encoder
			atten_out = self.dropout(atten_out)

		atten_out+=inputs
		atten_normalized = self.layer_norm(atten_out)

		if self.is_decoder and mode == 'MT':
			assert context is not None,"Decoder blocks require context"
			context_atten_out = self.self_context_atten(context,context,atten_normalized)
			context_atten_out = self.dropout(context_atten_out)
			context_atten_out+=atten_normalized
			atten_normalized = self.layer_norm(context_atten_out)

		ff_out=self.ff_layer(atten_normalized)
		ff_out = self.dropout(ff_out)
		ff_out+=atten_normalized
		ff_norm = self.layer_norm(ff_out)

		return tf.nn.relu(ff_norm)

class Position_Encoding_Layer(tf.keras.layers.Layer):
	def __init__(self, window_sz, emb_sz):
		super(Position_Encoding_Layer, self).__init__()
		self.positional_embeddings = self.add_weight("pos_embed",shape=[window_sz, emb_sz])

	@tf.function
	def call(self, x):
		"""
		Adds positional embeddings to word embeddings.
		:param x: [BATCH_SIZE x WINDOW_SIZE x EMBEDDING_SIZE ] the input embeddings fed to the encoder
		:return: [BATCH_SIZE x WINDOW_SIZE x EMBEDDING_SIZE ] new word embeddings with added positional encodings
		"""
		return x+self.positional_embeddings
