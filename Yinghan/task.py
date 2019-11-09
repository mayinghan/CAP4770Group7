############################
#      @author: Yinghan Ma
############################

import os
import sys
# sys.path.append("..")
# sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

import random
import numpy as np
import torch
import glob
import argparse
from heapq import *
from collections import OrderedDict

from data_utils import (convert_examples_to_features, logger, InputExample, InputFeatures, ClfProcessor)
from model import SentenceClf
from transformers import AdamW, WarmupLinearSchedule
from transformers import (WEIGHTS_NAME,
                                  BertConfig,
                                  BertForPreTraining,
                                  load_tf_weights_in_bert,
								  BertTokenizer)
from tqdm import tqdm, trange
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler

def err_handler(type, value, tb):
    logger.exception('Exception: {0}'.format(str(tb)))

sys.excepthook = err_handler


if __name__ == '__main__':
    print('success')