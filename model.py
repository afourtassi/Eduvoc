import gensim
from collections import defaultdict
from scipy import spatial
from gensim.models import KeyedVectors
from igraph import *
import random
import pandas as pd
from collections import Counter
"""
task
1. `model.py`: construct simple pipeline to build igraph, form df
2. `preprocess.py`: build several graphs (V,N) in ipython book
3. `visualize.py`: visualize network by sampling
"""

DIR = 'fb_data'
# paths to the 3 generated files from preprocess.py
BOOK_WORD_SET_FILE = '{}/book_words_set.txt'.format(DIR)
REBECCA_WORD_FILE = '{}/rebecca_words.txt'.format(DIR)
LEMMA_BOOK_FILE = "{}/cbt_lemma.txt".format(DIR)

def read_file_to_list(filename):
	words = []
	with open(filename, 'r') as f:
	    for word in f:
	        words.append(word.strip())
	return words

def getDistance(w1, w2, model):

    return abs(spatial.distance.cosine(model.wv[w1], model.wv[w2]))

def load_pretrained_word2vec(documents, size, min_count, iters, retrain=False):
	GOOGLE_W2V_FILE = './model/GoogleNews-vectors-negative300.bin'
	model = KeyedVectors.load_word2vec_format(GOOGLE_W2V_FILE, binary=True)  
	if not retrain:
		return model
	model_2 = gensim.models.Word2Vec(
	            documents,
	            sg=0,
	            size=size,
	            window=20,
	            min_count=min_count,
	            iter=iters,
	            workers=4)
	total_examples = model_2.corpus_count
	model_2.build_vocab([list(model.vocab.keys())], update=True)
	model_2.intersect_word2vec_format(GOOGLE_W2V_FILE, binary=True, lockf=1.0)
	model_2.train(documents, total_examples=total_examples, epochs=model_2.epochs)
	return model_2

def train_word2vec(documents, size=300, min_count=5, iters=100, window=20):
	model = gensim.models.Word2Vec(
        documents,
        sg=0,
        size=size,
        window=window,
        min_count=min_count,
        iter=iters,
        workers=4)
	model.train(documents, total_examples=len(documents), epochs=model.epochs)
	return model

def build_graph(model, sampling_size=0):

	book_set = read_file_to_list(BOOK_WORD_SET_FILE)
	edu_set = read_file_to_list(REBECCA_WORD_FILE)

	print("word2vec:\t", len(model.wv.vocab))
	print("book_set:\t", len(book_set))
	print("edu_set:\t", len(edu_set))
# 	surrounding_words = set(model.wv.vocab.keys()).intersection(book_set)
# 	surrounding_words = set(surrounding_words).difference(edu_set)
# 	print('surrounding words', len(surrounding_words))
	intersection_words = set(model.wv.vocab.keys()).intersection(edu_set)
	print('intersertion words', len(intersection_words))

	if sampling_size:
		token2vec = {}
		rand_words = random.sample(intersection_words, sampling_size)
		for word in rand_words:
			token2vec[word] = model.wv[word]
		for word in intersection_words:
			token2vec[word] = model.wv[word]
	else:
		token2vec = model.wv.vocab # bad

	idx = 0
	id2token = {}
	token2id = {}

	for word in token2vec:
	    id2token[idx] = word
	    token2id[word] = idx
	    idx += 1

	vertices = [idx for idx in range(len(token2vec))]
	edges = [(i, j) for i in vertices for j in vertices if i < j]
	g = Graph(vertex_attrs={"label":vertices}, edges=edges, directed=False)
	g.es["sim"] = [getSimilarity(id2token[i], id2token[j], model) for i,j in edges]
	g.es["dist"] = np.array(1-np.array(g.es['sim'])).tolist()
	# test validity
	# assert(getSimilarity('man', 'woman', model) == g[token2id['man'], token2id['woman']] )
	# assert(getSimilarity('cat', 'dog', model) == g[token2id['cat'], token2id['dog']] )

	return g, (edges, vertices), (id2token, token2id)

def filter_graph(weights, edges, vertices, id2token,  threshold=1.0):

	new_edges, new_weights = [], [] # weights: similarity
	new_distances = []
	for edge, w in zip(edges, weights):
	    if w >= threshold:
	        new_edges.append(edge)
	        new_weights.append(w)
	        new_distances.append(1-w)
	g0 = Graph(vertex_attrs={"label":vertices}, edges=new_edges, directed=False)

	g0.es["sim"] = new_weights
	g0.es["dist"] = new_distances
	g0.vs["label"] = [id2token[idx] for idx in vertices]

	return g0

def compute_measures_df(g, edges, vertices, id2token, token2id):
	"""
	compute centrality measures to output df
	"""

	# g -> strength, closeness (continuous)
	strengthRank = g.strength(None,  weights=g.es['sim'])
	closenessRank = g.closeness(None, 'all', weights=g.es['dist'], normalized=True)

	# g1 -> betweenness, eigen_centrality
	g1 = filter_graph(g.es["sim"], edges, vertices, id2token, threshold=0.5)
    
	eigen_centralityRank = g.eigenvector_centrality(directed=False, weights=g.es['sim'])

	betweennessRank = g.betweenness(vertices=None, 
                                     directed=False, 
                                     weights=g.es['dist'])

	# g2 -> degree
	g2 = filter_graph(g.es["sim"], edges, vertices, id2token, threshold=0.2)

	degreeRank = g2.degree(None, mode='all')
	print(degreeRank)

	# frequency
	wordCounter = defaultdict(int)
	with open(LEMMA_BOOK_FILE) as f:
	    for line in f:
	        for w in line.split():
	            w = w.strip()
	            if w in token2id:
	                wordCounter[w] += 1
	freqRank = [wordCounter[w] for w in token2id]

	# rebecca
	with open(REBECCA_WORD_FILE, 'r') as f:
	    edu_list = []
	    for word in f:
	        edu_list.append(word.strip())
	        
	rebeccaRank = {w:i+1 for i, w in enumerate(edu_list)}
	final_words_set = set(token2id.keys()).intersection(edu_list)

	print(len(strengthRank))
	print(len(betweennessRank))
	print(len(closenessRank))
	print(len(eigen_centralityRank))
	print(len(degreeRank))

	# create df
	data = []
	words_inorder = [id2token[idx] for idx in range(len(token2id))]
	for i, word in enumerate(words_inorder):
	    if word in final_words_set:
	        data.append([word,
	                    rebeccaRank[word],
	                    strengthRank[i], 
	                    closenessRank[i], 
	                    betweennessRank[i], 
	                    eigen_centralityRank[i],
	                    degreeRank[i],
	                    freqRank[i]])


	df = pd.DataFrame(data, columns=['word', 'ppvt', 'strgth', 
                                     'close', 'betw', 'eigen', 'degree', 'freq'])
	df = df.sort_values(by=['ppvt'])
	print(df.head())
	return df

def build_pos_graph(model, pos_word_set, sampling_size=0, max_dist=1):
	"""
	build semantic graph without referring Rebecca's word list;
	only to explore noun, verb's centrality in vocab
	"""

	book_set = read_file_to_list(BOOK_WORD_SET_FILE)
	edu_set = read_file_to_list(REBECCA_WORD_FILE)
	valid_pos_words = set(model.wv.vocab.keys()).intersection(edu_set)
	valid_pos_words = set(valid_pos_words).intersection(pos_word_set)
	# print('pos_word_set', len(pos_word_set))

	token2vec = {}
	if sampling_size > 0:
		rand_words = random.sample(valid_pos_words, sampling_size)
		for word in rand_words:
			token2vec[word] = model.wv[word]			
	else:
		token2vec = valid_pos_words

	idx = 0
	id2token = {}
	token2id = {}

	for word in token2vec:
	    id2token[idx] = word
	    token2id[word] = idx
	    idx += 1

	vertices = [idx for idx in range(len(token2vec))]
	edges = [(i, j) for i in vertices for j in vertices if i < j]
	weights = [getSimilarity(id2token[i], id2token[j], model) for i, j in edges]

	g = filter_graph(weights, edges, vertices, id2token, max_dist)

	return g, (edges, vertices), (id2token, token2id)

def read_docs():
	with open(LEMMA_BOOK_FILE) as f:
	    documents = []
	    for line in f:
	        documents.append(line.split())
	return documents

def freq_counts(documents):
    allwords = []
    for doc in documents:
        allwords += doc
    
    ct = Counter(allwords)
    for i in range(5):
        print('freq ', i+1, len([w for w in ct if ct[w] >= i+1]))














