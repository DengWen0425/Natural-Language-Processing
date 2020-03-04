from data_utils import *
from math import log10
from softmaxreg import accuracy
import numpy as np
import matplotlib.pyplot as plt


# Load the dataset
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)

# Load the train set
trainset = dataset.getTrainSentences()
nTrain = len(trainset)
trainLabels = np.zeros((nTrain,), dtype=np.int32)
for i in range(nTrain):
    words, trainLabels[i] = trainset[i]

# Load the dev set
devset = dataset.getDevSentences()
nDev = len(devset)
devLabels = np.zeros((nDev,), dtype=np.int32)
for i in range(nDev):
    words, devLabels[i] = devset[i]


# define a fun to deal negation
def deal_negation(doc):
    negation = False
    for i in range(len(doc)):
        if negation:
            if doc[i] not in [".", ",", ";", "?", "!", "..."]:
                doc[i] = "Not_" + doc[i]
            else:
                negation = False
        if doc[i] in ["not", "n't"]:  # to be added
            negation = True
    return doc


# define a fun to remove duplicates
def remove_duplicates(doc):
    temp = []
    for w in doc[0]:
        if w not in temp:
            temp.append(w)
    return temp


# define the bag of words model
class BagOfWords(object):

    def __init__(self):
        self.vocabulary = set()
        self.vocabularyNum = 0
        self.each_dic = [{} for i in range(5)]
        self.each_vocNum = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        self.each_label_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        self.negation_dealing = None
        self.boolean = None

        self.alpha = 1  # add-k parameter default value
        self.prior_prob = {}
        self.likelihood_prob = [{} for i in range(5)]

    def bag_construction(self, data, negation_dealing=True, boolean=False):
        self.negation_dealing = negation_dealing
        self.boolean = boolean
        for sent in data:
            label = sent[1]
            sent = sent[0]
            self.each_label_count[label] += 1

            if self.negation_dealing:
                sent = deal_negation(sent)

            if self.boolean:
                sent = remove_duplicates(sent)

            for word in sent:
                if word not in self.vocabulary:
                    self.vocabulary.add(word)
                    self.vocabularyNum += 1
                if word not in self.each_dic[label]:
                    self.each_dic[label][word] = 1
                    self.each_vocNum[label] += 1
                else:
                    self.each_dic[label][word] += 1

    def modify_alpha(self, alpha):
        self.alpha = alpha
        self.prior_prob = {}
        self.likelihood_prob = [{} for i in range(5)]

    def prior_p(self, label):
        if label not in self.prior_prob:
            self.prior_prob[label] = -log10(float(self.each_label_count[label])/sum(self.each_label_count.values()))
        return self.prior_prob[label]

    def likelihood_p(self, label, sent):
        result = 0.0
        for word in sent:
            if word not in self.likelihood_prob[label]:
                self.likelihood_prob[label][word] = -log10(
                    float(self.each_dic[label].get(word, 0) + self.alpha) / self.each_vocNum[label])
            result += self.likelihood_prob[label][word]
        return result

    def single_classify(self, sent):
        if self.negation_dealing:
            sent = deal_negation(sent)

        if self.boolean:
            sent = remove_duplicates(sent)

        all_prob = []
        for label in range(5):
            all_prob.append(self.prior_p(label)+self.likelihood_p(label, sent))

        return all_prob.index(min(all_prob))

    def batch_classify(self, data):
        result = np.zeros((len(data), ), dtype=np.int32)
        idx = 0
        for sent in data:
            sent = sent[0]
            result[idx] = self.single_classify(sent)
            idx += 1
        return result


# define a range of alpha
Alpha = [i/100.0 for i in range(1, 101, 2)]


# train a best alpha
def train_add_k_param(bag, dev, alphas, dev_labels):
    dev_accuracy = []
    for alpha in alphas:
        bag.modify_alpha(alpha)
        pred = bag.batch_classify(dev)
        dev_accuracy.append((accuracy(dev_labels, pred), alpha))

    # print(dev_accuracy)

    return [x[0] for x in dev_accuracy], max(dev_accuracy)[1]


if __name__ == "__main__":

    all_dev_acc = []
    all_test_acc = []
    for (negation, boolean) in [(False, False), (True, False), (False, True), (True, True)]:
        trainBags = BagOfWords()
        trainBags.bag_construction(trainset, negation, boolean)
        dev_acc, best_alpha = train_add_k_param(trainBags, devset, Alpha, devLabels)
        print("Best alpha value: %f" % best_alpha)
        trainBags.modify_alpha(best_alpha)

        # test data
        testset = dataset.getTestSentences()
        nTest = len(testset)
        testLabels = np.zeros((nTest,), dtype=np.int32)
        for i in range(nTest):
            words, testLabels[i] = testset[i]

        test_pred = trainBags.batch_classify(testset)
        test_acc = accuracy(testLabels, test_pred)

        all_test_acc.append(test_acc)
        all_dev_acc.append(dev_acc)

    for test_acc in all_test_acc:
       print("Test accuracy (%%): %f" % test_acc)

    for dev_acc in all_dev_acc:
        plt.plot(Alpha, dev_acc)
    plt.title("Accuracy on dev set")
    plt.xscale('log')
    plt.xlabel("alpha")
    plt.ylabel("accuracy")
    plt.legend(['neg=F, boo=F', 'neg=T, boo=F', 'neg=F, boo=T', 'neg=T, boo=T'], loc='upper left')
    plt.savefig("alpha_acc.png")
    plt.show()




