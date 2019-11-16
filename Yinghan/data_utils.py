############################
#      @author: Yinghan Ma
############################

import re
import torch
import random
import logging
import os
import sys
import string
import time

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

logfile = '../log/' + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) + '.log'
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    filename=logfile,
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s'))

logger = logging.getLogger()
logger.addHandler(console)

class InputExample(object):
    def __init__(self, guid, text, label):
        self.guid = guid
        self.text = text
        self.label = label

class InputFeatures(object):
    def __init__(self, input_ids, attention_masks, segment_ids, label):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.segment_ids = segment_ids
        self.label = label

def _readfile(filename):
    sentence = []
    score = []
    pack = []
    with open(filename) as f:
        for line in f:
            splits = line.split('\t')
            sentence.append(splits[0])
            score.append(1 if float(splits[1]) >= 0.5 else 0)
            pack.append((splits[0], 1 if float(splits[1]) >= 0.5 else 0))
    
    return pack

def convert_single_example(idx, example, max_seq_length, tokenizer, split_ratio=0.25):
    max_seq_length -= 2
    text = example.text
    label = example.label
    tokens = tokenizer.tokenize(text)
    split_point = int(split_ratio * max_seq_length)
    
    if(len(tokens) > max_seq_length):
        tokens = tokens[:split_point] + tokens[split_point - max_seq_length:]
        assert len(tokens) == max_seq_length
    
    tokens.insert(0, '[CLS]')
    tokens.append('[SEP]')
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(input_ids)
    
    max_seq_length += 2
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        attention_mask.append(0)
        
    
    segment_ids = [0] * len(input_ids)
    
    assert len(input_ids) == max_seq_length
    assert len(attention_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    
    if idx < 3:
        logger.info('**** Example *****')
        logger.info('guid %s' % (example.guid))
        logger.info('rawtext %s' % (example.text))
        logger.info('input_ids %s' % " ".join([str(x) for x in input_ids]))
        logger.info('masks %s' % " ".join([str(x) for x in attention_mask]))
        logger.info('segment_ids %s' % " ".join([str(x) for x in segment_ids]))
        logger.info('label %d', label)

    single_feature = InputFeatures(
        input_ids,
        attention_mask,
        segment_ids,
        label
    )
    
    return single_feature

def convert_examples_to_features(args, examples, tokenizer):
    features = []
    
    for i, example in enumerate(examples):
        feature = convert_single_example(i, example, args.max_seq_length, tokenizer, args.split_ratio)
        features.append(feature)
    
    return features

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return _readfile(input_file)
    
class ClfProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'train.csv')), "train")
    
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'dev.csv')), "dev")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'test.csv')), "test")
    
    def get_labels(self):
        return [0, 1]
            
    def _create_examples(self,lines,set_type):
        examples = []
        labels = []
        # for debug purpose
        # lines = random.sample(lines, int(len(lines) * 0.2))
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = sentence
            labels.append(label)
            examples.append(InputExample(guid=guid,text=text_a, label=label))
        return examples, labels
    
    
if __name__ == '__main__':
    pass