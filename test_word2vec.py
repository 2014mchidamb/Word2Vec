from word2vec import Word2Vec
import tensorflow as tf

f = open('corpus.txt', 'r')
corpus = f.read().lower().split()
f.close()

config = {}
config['corpus'] = corpus
config['window_size'] = 1
config['embed_size'] = 2
config['batch_size'] = 20
config['lrn_rate'] = 0.01
config['neg_sample_size'] = 10
config['model_type'] = 'skip-gram'
config['save_file'] = 'word2vec.ckpt'
config['num_iter'] = 3000

with tf.Session() as sess:
	w2v = Word2Vec(config, sess)
	w2v.train()
