import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re, string

# DIR = 'fb_data'
# BOOK_FILE = '{}/cbt.txt'.format(DIR)
# PPVT_LEMMA_FILE = '{}/PPVT_lemma.csv'.format(DIR)

def read_rebecca_lemma(PPVT_LEMMA_FILE):
    with open(PPVT_LEMMA_FILE) as f:
        lines = f.readlines()
        ranked_words = [line.strip() for line in lines]
    return ranked_words

def nltk2wn_tag(nltk_tag):
  if nltk_tag.startswith('J'):
    return wordnet.ADJ
  elif nltk_tag.startswith('V'):
    return wordnet.VERB
  elif nltk_tag.startswith('N'):
    return wordnet.NOUN
  elif nltk_tag.startswith('R'):
    return wordnet.ADV
  else:          
    return None

def lemmatize_sentence(sentence, lemmatizer, pos_words):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
    res_words = []
    for word, tag in wn_tagged:        
        # lemmatization
        if tag:
            word = lemmatizer.lemmatize(word, tag)
        res_words.append(word)
        if tag in pos_words:
            pos_words[tag].add(word)
    return " ".join(res_words)

def lemmatize_book(BOOK_FILE, LEMMA_BOOK_FILE):
    """
    returns book_words<list>, pos_words<dict<set>>
    """
    lemmatizer = WordNetLemmatizer()
    book_words = []
    pos_words = {wordnet.NOUN: set(), wordnet.VERB: set()}
    with open(BOOK_FILE) as f, open(LEMMA_BOOK_FILE, 'w') as out:
        for line in f:
            if len(line) > 1:
                regex = "[%s’]" % re.escape(string.punctuation)
                line = re.sub(regex, "", line).lower()
                line = lemmatize_sentence(line, lemmatizer, pos_words)
                book_words += line.split()
                # print(line)
                out.write(line)
    return book_words, pos_words

def generate_wordset_files(book_words, ranked_words, DIR):

    book_words_set = set(book_words)
    ranked_words_set = set(ranked_words)
    intersection_set = book_words_set.intersection(ranked_words_set)

    print(len(intersection_set))
    with open('{}/intersection.txt'.format(DIR), 'w') as out:
        for word in intersection_set:
            out.write(word)
            out.write('\n')

    print(len(book_words_set))
    with open('{}/book_words_set.txt'.format(DIR), 'w') as out:
        for word in book_words_set:
            out.write(word)
            out.write('\n')

    print(len(ranked_words_set))       
    with open('{}/rebecca_words.txt'.format(DIR), 'w') as out:
        for word in ranked_words:
            out.write(word)
            out.write('\n')

























