import os
import torch
import csv


from torch.utils.data import TensorDataset
from typing import List

LABEL2ID = {'0': 0, '1': 1}


class InputExample(object):
    """A single training/test example"""

    def __init__(self, nid, sent1, sent2, label):
        self.nid = nid
        self.sent1 = sent1
        self.sent2 = sent2
        self.label = label

    def __repr__(self):
        return str(self.__dict__)


class InputFeatures(object):
    """A single training/test example"""

    def __init__(self, sent1_id, sent2_id, label_id):
        self.sent1_id = sent1_id
        self.sent2_id = sent2_id
        self.label_id = label_id

    def __repr__(self):
        return str(self.__dict__)


class DataProcessor(object):
    """Processor for the Commonsense Validation and Explanation data set."""

    @classmethod
    def _read_tsv(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t")
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")))

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")))

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")))

    def get_labels(self, data_dir):
        """See base class."""
        return [0, 1]

    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[2]
            text_b = line[3]
            label = line[1]
            examples.append(
                InputExample(nid=guid, sent1=text_a, sent2=text_b, label=label))
        return examples


def convert_examples_to_features(examples: List[InputExample], label2id, max_seq_length, tokenizer) -> List[InputFeatures]:
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    pad_token = tokenizer.pad_token_id

    for (i, example) in enumerate(examples):
        sent1_tok = example.sent1
        sent2_tok = example.sent2
        sent1_id = tokenizer.encode(sent1_tok, add_special_tokens=True)
        sent2_id = tokenizer.encode(sent2_tok, add_special_tokens=True)

        assert len(sent1_id) < max_seq_length and len(
            sent2_id) < max_seq_length

        padding_length1 = max_seq_length - len(sent1_id)
        padding_length2 = max_seq_length - len(sent2_id)

        sent1_id = sent1_id + [pad_token] * padding_length1
        sent2_id = sent2_id + [pad_token] * padding_length2

        label_id = label2id[example.label]

        features.append(
            InputFeatures(
                sent1_id=sent1_id,
                sent2_id=sent2_id,
                label_id=label_id
            )
        )

    return features


def convert_features_to_dataset(features: List[InputFeatures]):

    all_sent1_id = torch.tensor(
        [f.sent1_id for f in features], dtype=torch.long)
    all_sent2_id = torch.tensor(
        [f.sent2_id for f in features], dtype=torch.long)
    all_label_id = torch.tensor(
        [f.label_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_sent1_id, all_sent2_id, all_label_id)
    return dataset


