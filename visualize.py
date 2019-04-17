from preprocess import *
from model import *

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

for i in range(10):
	max_dist = 0.1*(i+1)

	g = build_pos_graph(model, None, sampling_size=0, max_dist=max_dist)

	visual_style = {}
	visual_style["vertex_label_color"] = "#0088ff"
	visual_style["vertex_label_size"] = 15
	visual_style["vertex_size"] = 20
	visual_style["vertex_label"] = g.vs["label"]
	layout = g.layout("kk")
	visual_style["layout"] = layout
	visual_style["bbox"] = (300, 300)
	visual_style["margin"] = 20
	filename = "./output/rebec{}.pdf".format(i)
	plot(g, filename, **visual_style)

