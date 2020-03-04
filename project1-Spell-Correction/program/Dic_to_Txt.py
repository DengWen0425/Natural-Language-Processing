import nltk
from nltk.corpus import reuters
import re
from math import log10


# --------------- Unigram and bigram

categories = reuters.categories()
corpus = reuters.sents(categories=categories)

unigram = {}
bigram = {}

for sent in corpus:
    if sent[-1] == ".":
        sent[-1] = "</s>"
    else:
        sent = sent + ["</n>"]
    sent = ["<s>"] + sent
    for i in range(len(sent)-1):
        if sent[i] not in unigram:
            unigram[sent[i]] = 1
        else:
            unigram[sent[i]] += 1
        if (sent[i], sent[i+1]) not in bigram:
            bigram[sent[i], sent[i+1]] = 1
        else:
            bigram[(sent[i], sent[i+1])] += 1


uni_sum = sum(unigram.values())
V_uni = len(unigram)
bi_sum = sum(bigram.values())
V_bi = len(bigram)

f = open('unigram.txt', 'w')

f.write(str(unigram))

f.close()

f = open('bigram.txt', 'w')

f.write(str(bigram))

f.close()


