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
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from sklearn import metrics
import weightpath

MODEL_CLASSES = {
    'bert': (SentenceClf, BertTokenizer)
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def train(args, train_dataset, model, tokenizer):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

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
    best_f1 = 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc='Epoch', disable=args.local_rank not in [-1, 0])

    for _ in train_iterator:
        epoch_loss = 0.0
        epoch_iteration = tqdm(train_dataloader, desc='Iter')
        for step, batch in enumerate(epoch_iteration):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'segment_ids': batch[2]
            }
            y_true = batch[3].view(-1, 1).float()
            
            logits = model(**inputs)
            # print(logits.type())
            # print(y_true.type())
            loss_func = BCEWithLogitsLoss()
            loss = loss_func(logits, y_true)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            epoch_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
        
        logger.info('Training loss current epoch: %f', epoch_loss)
    
    return global_step, tr_loss / global_step

def evaluate(args, test_dataset, model, tokenizer, mode='test'):
    pass

def load_and_cache_examples(args, processor, tokenizer, evaluate=False, dev=False, output_examples=False):
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}'.format(
        'eval' if evaluate else 'train',
        list(filter(None, args.bert_model.split('/'))).pop(),
        str(args.max_seq_length)
    ))
    
    # TODO: refactor the Example structure, save both Example and labels to cache
    if os.path.exists(cached_features_file) and not args.overwrite_cache and not output_examples:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        # label_list = processor.get_labels()

        examples, labels = processor.get_test_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(args,
                                                examples, 
                                                tokenizer)
        
        logger.info('saving features into cache file %s', cached_features_file)
        torch.save(features, cached_features_file)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_masks for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    # TODO: check shape
    dataset = (all_input_ids, all_attention_masks, all_segment_ids, all_labels)

    return dataset

def main():
    parser = argparse.ArgumentParser()

    # required parameters
    parser.add_argument('--data_dir', 
                        default=None,
                        type=str,
                        required=True,
                        help="Dir of input data. DON'T include exact file name")
    parser.add_argument('--bert_model', default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument('--task',
                        default=None,
                        type=str,
                        required=True,
                        help='Will use as name of saved models and result file')
    parser.add_argument('--output_dir',
                        default=None,
                        type=str,
                        required=True)
    
    # Other optional parameters
    parser.add_argument("--model_type", 
                        default='bert')
    parser.add_argument("--split_ratio",
                        default=0.25,
                        type=float)
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', 
                        action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', 
                        action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_ratio", default=0.0, type=float,
                        help="Linear warmup over warmup_ratio.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--dev_step',
                        type=float, default=1000)
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()


    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup CUDA
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    args.device = device

    # Set seed
    set_seed(args)
    # args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES['bert']
    processor = ClfProcessor()
    # label_list = processor.get_labels()
    # num_labels = len(processor.get_labels())

    # initialize tokenizer and model from the downloaded tf checkpoint
    if args.bert_model == 'bert-base-cased':
        vocab_file = weightpath.BASE_VOCAB_FILE
        config_file = weightpath.BASE_CONFIG_FILE
        weight_file = weightpath.BASE_WEIGHTS
    elif args.bert_model == 'wwm':
        vocab_file = weightpath.WWM_VOCAB_FILE
        config_file = weightpath.WWM_CONFIG_FILE
        weight_file = weightpath.WWM_WEIGHTS
    else:
        raise ValueError('Currently only support Bert Base Cased(bert-base-cased) and Whole Word Masking Cased(wwm)')
    
    # prepare the pretrained model and tokenizer
    tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=False)
    config = BertConfig.from_pretrained(config_file)
    model = model_class.from_pretrained('bert-base-cased')

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train_dataset = TensorDataset(*load_and_cache_examples(args, processor, tokenizer))
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

if __name__ == '__main__':
    main()