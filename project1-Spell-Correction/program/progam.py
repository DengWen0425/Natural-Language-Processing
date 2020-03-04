# Implement a language model.

import re
from math import log10
import datetime


# --------------- Read the vocabulary

vocab = set([line.rstrip() for line in open('vocab.txt')])

# --------------- Unigram and bigram

f = open('unigram.txt', 'r')
unigram = eval(f.read())
f.close()
f = open('bigram.txt', 'r')
bigram = eval(f.read())
f.close()

uni_sum = sum(unigram.values())
V_uni = len(unigram)
bi_sum = sum(bigram.values())
V_bi = len(bigram)

# --------------- Channel probability

spell_errors = open("count_1edit.txt", "r").readlines()

spell_errors = [x.strip().split("\t") for x in spell_errors]

channel_prob = {}

for item in spell_errors:
    channel_prob[item[0]] = int(item[1])

eSum = sum(channel_prob.values())

# ---------------  Generate candidate words


# 1 edit distance words
def edit1_words(word):
    alphabet = "abcdefghijklmnopqrstuvwxyz"

    spilt = [(word[:i], word[i:]) for i in range(len(word)+1)]

    result = {}

    for left, right in spilt:
        if len(left) > 0:
            prev_letter = left[-1]
        else:
            prev_letter = ">"
        # delete
        if len(right) != 0:
            result[left+right[1:]] = prev_letter+right[0]+"|"+prev_letter
        # transposition
        if len(right) > 1:
            result[left+right[1]+right[0]+right[2:]] = right[0]+right[1]+"|"+right[1]+right[0]
        for c in alphabet:
                # Insert
                result[left+c+right] = prev_letter+"|"+prev_letter+c
                # Subs
                if len(right) != 0:
                    result[left+c+right[1:]] = right[0]+"|"+c

    return result


# words that in the vocab
def candidate_words(edit_words):
    return {word: error for word, error in edit_words.items() if word in vocab}


# 2 edit distance words
def edit2_words(word_errors):
    alphabet = "abcdefghijklmnopqrstuvwxyz"

    result = {}

    for word, error in word_errors.items():
        spilt = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        for left, right in spilt:
            if len(left) > 0:
                prev_letter = left[-1]
            else:
                prev_letter = ">"
            # delete
            if len(right) != 0:
                result[left+right[1:]] = error + "+" + prev_letter+right[0]+"|"+prev_letter
            # transposition
            if len(right) > 1:
                result[left+right[1]+right[0]+right[2:]] = error + "+" + right[0]+right[1]+"|"+right[1]+right[0]
            for c in alphabet:
                # insert
                result[left+c+right] = error + "+" + prev_letter+"|"+prev_letter+c
                # delete
                if len(right) != 0:
                    result[left+c+right[1:]] = error + "+" + right[0]+"|"+c

    return result


# --------------- Data processing
special_w = {"<s>", "</s>", "s'", "n't", "'s", "'d", "'ll", "'ve", "'re", ",", "?", "</n>"}


def data_processing(testData):
    for i in range(len(testData)):

        if testData[i][2][-1] == ".":
            sent = ["<s>"] + testData[i][2][:-1].split() + ["</s>"]
        else:
            sent = ["<s>"] + testData[i][2].split() + ["</n>"]
        j = 0
        while j < len(sent):

            # Deal with the form like "year's"
            if sent[j][-2:] in ["'s", "'d", "s'"]:
                sent.insert(j+1, sent[j][-2:])
                sent[j] = sent[j][:-2]
                j += 2
                continue

            # Deal with the form like "we'll"
            if sent[j][-3:] in ["'ll", "'ve", "'re", "n't"]:
                sent.insert(j + 1, sent[j][-3:])
                sent[j] = sent[j][:-3]
                j += 2
                continue

            # Deal with the form like "word,"
            if sent[j][-1] in [",", "?"]:
                sent.insert(j + 1, sent[j][-1])
                sent[j] = sent[j][:-1]
                j += 2
                continue

            # if sent[j][-1] == ".":
                # sent.insert(j + 1, sent[j][-1])
                # sent[j] = sent[j][:-1]
                # j += 2
                # continue
            j += 1
        # Save
        testData[i][2] = sent
    return testData


# --------------- Correction:

"""
# add-1 smoothing
def prob_add1(pre_w, candidate, back_w, ori_w):
    word, edit = candidate
    if ori_w in vocab and word == ori_w:
        cp = -log10(0.95)
    else:
        if "+" in edit:
            tmp = edit.split("+")
            cp = -log10(channel_prob.get(tmp[0], 1)/float(eSum)) - log10(channel_prob.get(tmp[1], 1)/float(eSum))
        else:
            cp = -log10(channel_prob.get(edit, 1)/float(eSum))
    return -log10(bigram.get((pre_w, word), 1) / (float(bi_sum) + V_bi)) - log10(
        bigram.get((word, back_w), 1) / (float(bi_sum) + V_bi)) + cp
"""


# Interpolation
def prob_interpolation(pre_w, candidate, back_w, ori_w):
    word, edit = candidate
    if ori_w in vocab and word == ori_w:
        cp = -log10(0.95) - log10(unigram.get(word, 1)/uni_sum)
    else:
        if "+" in edit:
            tmp = edit.split("+")
            cp = -log10(channel_prob.get(tmp[0], 1) / float(eSum)) - log10(
                channel_prob.get(tmp[1], 1) / float(eSum)) - log10(unigram.get(word, 1)/uni_sum)
        else:
            cp = -log10(channel_prob.get(edit, 1) / float(eSum)) - log10(unigram.get(word, 1)/uni_sum)

    return -log10(
        0.99999999 * bigram.get((pre_w, word), 0) / float(bi_sum) + 0.00000001 * unigram.get(back_w, 1) / float(
            uni_sum)) - log10(
        0.99999999 * bigram.get((word, back_w), 0) / float(bi_sum) + 0.00000001 * unigram.get(back_w, 1) / float(
            uni_sum)) + cp


"""
# good turing
bi_count = {}
for i in range(22):
    bi_count[i] = len([x for x in bigram.values() if x == i])


def prob_good_turing(pre_w, candidate, back_w, ori_w):
    word, edit = candidate
    if ori_w in vocab and word == ori_w:
        cp = -log10(0.95)
    else:
        if "+" in edit:
            tmp = edit.split("+")
            cp = -log10(channel_prob.get(tmp[0], 1) / float(eSum)) - log10(channel_prob.get(tmp[1], 1) / float(eSum))
        else:
            cp = -log10(channel_prob.get(edit, 1) / float(eSum))

    def bigram_prob(bi):
        if bi not in bigram:
            return bi_count[1]/float(V_bi)
        else:
            r = bigram[bi]
            if r <= 20:
                return r * (bi_count[r+1]/bi_count[r])/float(V_bi)
            else:
                return bigram[bi] / float(bi_sum)

    return -log10(bigram_prob((pre_w, word))) - log10(bigram_prob((word, back_w))) + cp



def prob_back_off(pre_w, candidate, back_w, ori_w):
    word, edit = candidate
    if ori_w in vocab and word == ori_w:
        cp = -log10(0.95) - log10(unigram.get(word, 1) / uni_sum)
    else:
        if "+" in edit:
            tmp = edit.split("+")
            cp = -log10(channel_prob.get(tmp[0], 1) / float(eSum)) - log10(
                channel_prob.get(tmp[1], 1) / float(eSum)) - log10(unigram.get(word, 1)/uni_sum)
        else:
            cp = -log10(channel_prob.get(edit, 1) / float(eSum)) - log10(unigram.get(word, 1)/uni_sum)

    def bigram_prob(bi):
        if bi not in bigram:
            return unigram.get(bi[1], 1) / float(uni_sum)
        else:
            return bigram.get(bi) / float(uni_sum)

    return -log10(bigram_prob((pre_w, word))) - log10(bigram_prob((word, back_w))) + cp
"""


# Begin to correct
def correction(testData, smooth_method=prob_interpolation):
    with open("result.txt", "w") as output:
        def isNumber(s):
            if len(re.findall(r"\d+(\.\d+)?", s)) > 0:
                return True
            return False

        for i in range(len(testData)):
            sent = testData[i][2]
            c_count = 0
            output.write(str(i)+"\t")

            # Non-word error
            for j in range(len(sent)):
                if sent[j] not in vocab and sent[j] not in special_w and sent[j] != "":
                    edit1 = edit1_words(sent[j])
                    candidates = candidate_words(edit1)
                    edit2 = edit2_words(edit1)
                    for w, e in candidate_words(edit2).items():
                        if w not in candidates:
                            candidates[w] = e

                    if len(candidates) == 0:
                        continue
                    sent[j] = min(candidates.items(), key=lambda x: smooth_method(sent[j-1], x, sent[j+1], sent[j]))[0]
                    c_count += 1

            # real-word error
            if c_count < int(testData[i][1]):
                for j in range(len(sent)):
                    if sent[j] not in special_w and sent[j] != "" and not isNumber(sent[j]):
                        edit1 = edit1_words(sent[j])
                        candidates = candidate_words(edit1)
                        if len(candidates) == 0:
                            edit2 = edit2_words(edit1)
                            candidates = candidate_words(edit2)
                        if len(candidates) == 0:
                            continue
                        sent[j] = min(candidates.items(),
                                      key=lambda x: smooth_method(sent[j - 1], x, sent[j + 1], sent[j]))[0]

            # output the ans
            for w in sent:

                if w == "<s>":
                    continue
                elif w == "</s>":
                    output.write(".")
                elif w == "</n>":
                    continue
                elif w in special_w:
                    output.write(w)
                else:
                    output.write(" " + w)
            output.write("\n")
        print("Done!")


if __name__ == "__main__":
    begin = datetime.datetime.now()

    testData = open("testdata.txt", "r").readlines()

    testData = [x.strip().split("\t") for x in testData]

    testData = data_processing(testData)

    correction(testData, prob_interpolation)

    print((datetime.datetime.now()-begin).seconds)


