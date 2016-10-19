from math import sqrt
import collections
import numpy as np
import tensorflow as tf

class Word2Vec(object):
	lrn_rate = 0.01
	model_type = 'skip-gram'
	window_size = 1
	embed_size = 2
	batch_size = 20
	neg_sample_size = 10
	chkpt_file = 'word2vec.ckpt'
	num_iter = 3000

	def __init__(self, config, sess):
		# The tensorflow session
		self.sess = sess
		# The corpus
		self.corpus = config['corpus']
		# Learning rate for optimization
		self.lrn_rate = config['lrn_rate']
		# Skip-gram or CBOW
		self.model_type = config['model_type']
		# Size of context
		self.window_size = config['window_size']
		# Size of embedding
		self.embed_size = config['embed_size']
		# Batch size
		self.batch_size = config['batch_size']
		# For NCE loss
		self.neg_sample_size = config['neg_sample_size']
		# For saving model
		self.chkpt_file = config['save_file']
		# Number of iterations
		self.num_iter = config['num_iter']

	def build_vocab(self):
		word_freq = collections.Counter(self.corpus).most_common()	
		self.idx_to_word = [wpair[0] for wpair in word_freq]
		self.word_to_idx = {w: i for i, w in enumerate(self.idx_to_word)}
		self.vocab_size = len(self.idx_to_word)
	
	def build_data(self):
		self.data = [self.word_to_idx[word] for word in self.corpus]

	def get_context_data(self):
		context_data = []
		for i in range(self.window_size, len(self.data)-self.window_size):
			context_data.append([self.data[i], self.data[i-self.window_size:i]+self.data[i+1:i+self.window_size+1]])
		self.context_data = context_data

	def get_skip_gram_data(self):
		skip_gram_data = []
		for p in self.context_data:
			for y in p[1]:
				skip_gram_data.append([p[0], [y]])
		self.skip_gram_data = skip_gram_data

	def gen_batch(self, size):
		x_data=[]
		y_data = []
		r = np.random.choice(range(len(self.skip_gram_data)), min(size, len(self.skip_gram_data)), replace=False)
		for i in r:
			x_data.append(self.skip_gram_data[i][0])
			y_data.append(self.skip_gram_data[i][1])
		return x_data, y_data				

	def build_skip_gram(self):
		# Skip-gram input: batch of indices for source words, batch of indices for targets
		self.train_inputs = tf.placeholder(tf.int32, [self.batch_size])
		self.train_labels = tf.placeholder(tf.int32, [self.batch_size, 1])
		# Skip-gram parameters: embedding, hidden layer matrix
		self.embeddings = tf.Variable(
			tf.random_uniform([self.vocab_size, self.embed_size], -1.0, 1.0))
		self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
		self.hidden_weights = tf.Variable(
			tf.truncated_normal([self.vocab_size, self.embed_size],
								stddev = 1.0/sqrt(self.embed_size)))
		self.hidden_biases = tf.Variable(tf.zeros([self.vocab_size]))
		# Compute noise-contrastive estimation loss
		self.loss = tf.reduce_mean(
			tf.nn.nce_loss(self.hidden_weights, self.hidden_biases, self.embed, self.train_labels,
				self.neg_sample_size, self.vocab_size))
		# Optimize
		self.optimizer = tf.train.AdamOptimizer(self.lrn_rate).minimize(self.loss)
		# Initialize params	
		self.sess.run(tf.initialize_all_variables())
		# Save embeddings
		self.saver = tf.train.Saver(var_list={"embeddings": self.embeddings})		

	def train(self):
		self.build_vocab()
		self.build_data()
		self.get_context_data()
		self.get_skip_gram_data()
		self.build_skip_gram()
		for step in range(self.num_iter):
			batch_inputs, batch_labels = self.gen_batch(self.batch_size)
			_, loss_val = self.sess.run([self.optimizer, self.loss],
				feed_dict={self.train_inputs: batch_inputs, self.train_labels: batch_labels})
			if step % 500 == 0:
				print("Loss at %d: %.5f" % (step, loss_val))
		self.saver.save(self.sess, self.chkpt_file)		
	
