from collections import defaultdict
import pickle  # To store the model to avoid repeat computation


class HMM(object):  # A model class
    def __init__(self):
        self.states = []
        self.observations = []
        self.initials = defaultdict(float)
        self.transitions = defaultdict(float)
        self.emissions = defaultdict(float)
        self.alpha = 1e-7  # This parameter is to do add-alpha smoothing later for emissions
        self.beta = 0  # This parameter is to do add-alpha smoothing later transitions

    def train(self, filename):
        with open(filename, "r", encoding="UTF-8") as train_file:
            train_set = train_file.readlines()
            begin = True
            count = 0
            pre_count = defaultdict(float)  # count for (prev_state, curr_state) pair
            state_count = defaultdict(float)  # count for (state, observation) pair
            for i in range(len(train_set)):
                item = train_set[i]
                if item == "\n":
                    count += 1
                    begin = True
                    continue
                item = item.rstrip().split("\t")
                if begin:
                    self.initials[item[1]] += 1
                    begin = False
                else:
                    prev = train_set[i-1].rstrip().split("\t")[1]
                    self.transitions[(prev, item[1])] += 1
                    pre_count[prev] += 1
                self.emissions[(item[1], item[0])] += 1
                state_count[item[1]] += 1
                if item[0] not in self.observations:
                    self.observations.append(item[0])
                if item[1] not in self.states:
                    self.states.append(item[1])

            for init in self.initials:
                self.initials[init] /= count

            for prev, curr in self.transitions:
                self.transitions[(prev, curr)] /= pre_count[prev]

            for state, observe in self.emissions:
                self.emissions[(state, observe)] /= state_count[state]
        return


# This is a function to read the model. If there hasn't trained a model, them it will train one and store it.
def model_read():
    arg_file = "argument_train.txt"
    trigger_file = "trigger_train.txt"

    try:
        with open("./Trigger_Model.pickle", "rb") as m1:
            TriggerHMM = pickle.load(m1)
        with open("./Arg_Model.pickle", "rb") as m2:
            ArgHMM = pickle.load(m2)
    except IOError:
        ArgHMM = HMM()
        ArgHMM.train(arg_file)
        TriggerHMM = HMM()
        TriggerHMM.train(trigger_file)
        with open("Trigger_Model.pickle", "wb") as m1:
            pickle.dump(TriggerHMM, m1)
        with open("Arg_Model.pickle", "wb") as m2:
            pickle.dump(ArgHMM, m2)

    return ArgHMM, TriggerHMM


class PredictionAlg(object):
    """
    This class is help us to do prediction
    """

    def __init__(self):
        self.test_set = []  # list with observation items
        self.pred = []  # list to store our prediction
        self.answer = []  # list to store the true answer

    def test_read(self, filename):
        """
        Fetch the test data and store them respectively
        """
        with open(filename, "r", encoding="UTF-8") as file:
            test = file.readlines()
            for item in test:
                item = item.rstrip()
                if item == "":  # End of an observation sequence
                    self.test_set.append(item)
                    self.answer.append(item)
                    continue
                self.test_set.append(item.split("\t")[0])  # observation
                self.answer.append(item.split("\t")[1])  # true answer

    def greedy_decode(self, HMM):

        def max_greedy_state(observe, prev=None):
            """
            Given observation and prev state
            return the state for the observation with max probability
            """
            if prev is None:
                return \
                    max([(prob * (HMM.emissions[(state, observe)] + HMM.alpha), state) for state, prob in
                         HMM.initials.items()])[1]
                # Note that add-alpha is to do smoothing that avoid 0 probability
            else:
                return max([((HMM.emissions[(state, observe)] + HMM.alpha) * (
                            HMM.beta + HMM.transitions[(prev, state)]), state) for state in HMM.states])[1]
                # Note that add-alpha is to do smoothing that avoid 0 probability

        Begin = True  # bool value stands for the beginning of the sentence
        self.pred = ["" for i in range(len(self.test_set))]
        for i in range(len(self.test_set)):
            item = self.test_set[i]
            if item == "":  # End of an observation sequence
                self.pred[i] = item
                Begin = True
                continue
            if Begin:  # Beginning of an observation sequence
                self.pred[i] = max_greedy_state(item)
                Begin = False
                continue
            prev = self.pred[i - 1]
            self.pred[i] = max_greedy_state(item, prev)

    def viterbi_decode(self, HMM):

        def backward(memory, idx):
            """
            backtrack the memory to get the optimal state sequence and store them
            """
            while memory:
                idx -= 1
                dic = memory.pop()
                self.pred[idx] = max([(v, s) for s, v in dic.items()])[1]

        Begin = True  # bool value stands for the beginning of the sentence
        self.pred = ["" for i in range(len(self.test_set))]
        for i in range(len(self.test_set)):
            item = self.test_set[i]
            if item == "":  # if reach the end of sequence then begin to backtrack to predict
                backward(Memory, i)
                self.pred[i] = item
                Begin = True
                continue
            if Begin:  # Beginning of an observation sequence
                count = 0
                Memory = [{}]  # to store the dynamic information in each stage
                for state in HMM.states:
                    Memory[count][state] = HMM.initials[state] * (HMM.emissions[(state, item)] + HMM.alpha)
                    # Note that add-alpha is to do smoothing that avoid 0 probability
                Begin = False
                continue
            count += 1
            Memory.append({})
            for state in HMM.states:
                Memory[count][state] = max(
                    [Memory[count - 1][prev] * (HMM.transitions[(prev, state)] + HMM.beta) for prev in
                     HMM.states]) * (HMM.emissions[(state, item)] + HMM.alpha)
                # Note that add-alpha is to do smoothing that avoid 0 probability

    def solve(self, HMM, method):
        if method == "greedy":
            self.greedy_decode(HMM)
        if method == "viterbi":
            self.viterbi_decode(HMM)


def pred2file(method):
    """
    :param method:
    :return: result by corresponding method (greedy HMM or viterbi HMM)
    """
    # Fetch the trained HMM model
    ArgHMM, TriggerHMM = model_read()

    # Argument
    ArgHMM.beta = 1/len(ArgHMM.states)  # to do smoothing
    ArgHMM.alpha = 1e-16
    ArgPred = PredictionAlg()
    ArgPred.test_read("argument_test.txt")
    ArgPred.solve(ArgHMM, method)

    with open("argument_result.txt", "w", encoding="UTF-8") as out:
        for i in range(len(ArgPred.test_set)):
            if ArgPred.test_set[i] == "":
                out.write("\n")
                continue
            out.write(ArgPred.test_set[i] + "\t" + ArgPred.answer[i] + "\t" + ArgPred.pred[i] + "\n")

    # TriggerHMM.beta seems to work better with no smoothing operation after plenty of experiments
    TriggerHMM.beta = 0
    TriggerHMM.alpha = 8e-6
    TriggerPred = PredictionAlg()
    TriggerPred.test_read("trigger_test.txt")
    TriggerPred.solve(TriggerHMM, method)

    with open("trigger_result.txt", "w", encoding="UTF-8") as out:
        for i in range(len(TriggerPred.test_set)):
            if TriggerPred.test_set[i] == "":
                out.write("\n")
                continue
            out.write(TriggerPred.test_set[i] + "\t" + TriggerPred.answer[i] + "\t" + TriggerPred.pred[i] + "\n")


if __name__ == '__main__':
    from eval import evaluation
    # pred2file("greedy")
    # pred2file("viterbi")

    pred2file("greedy")

    evaluation('trigger')
    evaluation('argument')

    pred2file("viterbi")

    evaluation('trigger')
    evaluation('argument')




