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
from sklearn import metrics

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def train(agrs, train_dataset, model, tokenizer):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(trian_dataset, sampler=train_sampler, batch_size=agrs.train_batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    if args.warmup_ratio:
        ws = args.warmup_ratio * t_total
    else:
        ws = args.warmup_steps

    # prepare the optimizer and scheduler (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=ws, t_total=t_total)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    candidate_f1 = []
    best_f1 = 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc='Epoch', disable=args.local_rank not in [-1, 0])

    for _ in train_iterator:
        epoch_loss = 0.0
        epoch_iteration = tqdm(train_dataloader, desc='Iter')
        for step, batch in enumerate(epoch_iteration):
            model.train()
            batch = tuple(t.to(agrs.device) for t in batch)
            
            # TODO: Backward propagation
            pass

def evaluate(args, test_dataset, model, tokenizer, mode='test'):
    pass

def load_and_cache_examples(args, processor, tokenizer, evaluate=False, dev=False, output_examples=False):
    pass



if __name__ == '__main__':
    print('success')