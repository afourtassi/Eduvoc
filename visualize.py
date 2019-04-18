from preprocess import *
from model import *
import numpy as np
# ------------------------ PREPROCESS -------------------------------
ranked_words= read_rebecca_lemma()

book_words, pos_words = lemmatize_book()

print('n\t', len(pos_words['n']))
print('v\t', len(pos_words['v']))

# generate_wordset_files(book_words, ranked_words)

# ---------------------TRAIN WORD2VEC -------------------------------
documents = read_docs()
# no pretrained
model = train_word2vec(documents, 300, 5, 200)
# use pretrained model
# model = load_pretrained_word2vec(documents, 300, 5, 100, retrain=False)
# use pretrained + retrain
# model = load_pretrained_word2vec(documents, 300, 5, 30, retrain=True)

freq_counts(documents)
print('man', 'woman', getDistance('man', 'woman', model))
print('cat', 'dog', getDistance('cat', 'dog', model))
print('sun', 'kid', getDistance('sun', 'kid', model))        

# ------------------------------BUILD + VISUALIZE GRAPH---------------------------
# g = build_pos_graph(model, pos_words['n'], sampling_size=50, max_dist=0.1) #change edu_set to pos_set


def normalize_list(l, lower, upper):
	max_min_range = max(l)-min(l)
	print('times', (upper-lower)/max_min_range)
	return [lower + (upper - lower) / max_min_range * x for x in l]

for i in range(10):
	max_dist = 0.1*(i+1)

	g, (edges, vertices), (id2token, token2id) = build_pos_graph(model, pos_words['n'], sampling_size=50, max_dist=max_dist)
	# rank = g.strength(None,  weights=g.es['weight'])
	# rank = g.closeness(None, 'all', weights=g.es['weight'], normalized=True)
	# g1 = filter_graph(g.es["weight"], edges, vertices, id2token, max_dist=0.5)
	# rank = g1.betweenness(directed=False, weights=g1.es['weight'])
	# rank = g1.eigenvector_centrality(directed=False, weights=g1.es['weight'])
	# g2 = filter_graph(g.es["weight"], edges, vertices, id2token, max_dist=0.1)
	# rank = g2.degree(mode='all')
 
	wordCounter = defaultdict(int)
	with open(LEMMA_BOOK_FILE) as f:
	    for line in f:
	        for w in line.split():
	            w = w.strip()
	            if w in token2id:
	                wordCounter[w] += 1
	rank = [wordCounter[w] for w in token2id]

	print('mean', np.mean(rank))
	vertexSize = normalize_list(rank, 10, 40)
	vertexLabelSize = normalize_list(rank, 20, 40)

	visual_style = {}
	visual_style["vertex_color"] = "#94a1b6"
	visual_style["vertex_label_color"] = "#f20606"
	visual_style["vertex_label_size"] = vertexLabelSize # 15
	visual_style["vertex_size"] = vertexSize # 15
	visual_style["vertex_label"] = g.vs["label"]
	layout = g.layout("kk")
	visual_style["layout"] = layout
	visual_style["bbox"] = (1000, 1000)
	visual_style["margin"] = 20
	filename = "./output/freq{}.pdf".format(i)
	plot(g, filename, **visual_style)
	break

