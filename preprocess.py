import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re, string


def read_rebecca_lemma():
    with open('data/PPVT_lemma.csv') as f:
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

def lemmatize_book():
    """
    returns book_words<list>, pos_words<dict<set>>
    """
    lemmatizer = WordNetLemmatizer()
    book_words = []
    pos_words = {wordnet.NOUN: set(), wordnet.VERB: set()}
    with open('data/book.txt') as f, open('data/lemma_book.txt', 'w') as out:
        for line in f:
            if len(line) > 1:
                regex = "[%s’]" % re.escape(string.punctuation)
                line = re.sub(regex, "", line).lower()
                line = lemmatize_sentence(line, lemmatizer, pos_words)
                book_words += line.split()
                # print(line)
                out.write(line)
    return book_words, pos_words

def generate_wordset_files(book_words, ranked_words):

    book_words_set = set(book_words)
    ranked_words_set = set(ranked_words)
    intersection_set = book_words_set.intersection(ranked_words_set)

    print(len(intersection_set))
    with open('data/intersection.txt', 'w') as out:
        for word in intersection_set:
            out.write(word)
            out.write('\n')

    print(len(book_words_set))
    with open('data/book_words_set.txt', 'w') as out:
        for word in book_words_set:
            out.write(word)
            out.write('\n')

    print(len(ranked_words_set))       
    with open('data/rebecca_words.txt', 'w') as out:
        for word in ranked_words:
            out.write(word)
            out.write('\n')

























