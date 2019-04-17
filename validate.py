from model import *

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
        
# build graph     
g, (edges, vertices), (id2token, token2id) = build_graph(model, sampling=True)

# compute measures
df = compute_measures_df(g, edges, vertices, id2token, token2id)

# see correlations
for col in df.columns[2:]:  
    print('rebecca vs. {}'.format(col), df['rebec'].corr(df[col]))

# save to csv
df.to_csv("./ranks_04_16_continuous_close_strgth.csv",index=False)