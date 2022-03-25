# the tokenization code is inspired from the huggingface example on question-answering task: https://github.com/huggingface/transformers/blob/master/examples/legacy/question-answering/run_squad.py  
import sys
sys.path.append('../')

import glob
import logging
import os
import random
import timeit

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import transformers
from transformers import AutoTokenizer, squad_convert_examples_to_features

from transformers.data.processors.squad import SquadResult, SquadV1Processor
from transformers.trainer_utils import is_main_process

import src.configs as configs

logger = logging.getLogger(__name__)


def load_and_cache_examples(domain_name, train_file, tokenizer, evaluate=False, output_examples=False):
    if configs.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = configs.output_dir if configs.output_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}_{}_{}".format(
            "train",
            list(filter(None, configs.original_model_name_or_path.split("/"))).pop(),
            str(configs.max_seq_length),
            "train" if configs.do_train else exit(1),
            domain_name,
        ),
    )

    print(cached_features_file)
    
    # init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not configs.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        processor = SquadV1Processor()
        examples = processor.get_dev_examples(configs.data_dir, filename=train_file)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=configs.max_seq_length,
            doc_stride=configs.doc_stride,
            max_query_length=configs.max_query_length,
            is_training=False,
            return_dataset="pt",
            threads=configs.threads,
        )

        if configs.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if configs.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset


def feature_to_folder(dataset_name, prefix, out_ft, out_pair): 
    cached_features_file = os.path.join(
          configs.output_dir,
          "cached_{}_{}_{}_{}".format(
              "train",
              list(filter(None, configs.original_model_name_or_path.split("/"))).pop(),
              str(configs.max_seq_length),
              str(dataset_name),
          )
    )

    features_and_dataset = torch.load(cached_features_file)
    features, dataset, examples = (
      features_and_dataset["features"],
      features_and_dataset["dataset"],
      features_and_dataset["examples"],
    )
  
    qid_list = []
    for i in range(len(features)):
        qid = features[i].__dict__['qas_id']
        uid = features[i].__dict__['unique_id']
        qid_list.append(f'{prefix}_{uid}')
  
    file = open(out_pair+f'{prefix}_qid_list.txt', 'w')
    for x in qid_list:
        file.write(f'{x}\n')
    file.close()

    for i in range(len(qid_list)):

        if i%5_000==0:
            print(f'i: {i}')

        feat = features[i]

        input_ids = feat.input_ids
        attention_masks = feat.attention_mask
        token_type_ids = feat.token_type_ids
        start_positions = feat.start_position
        end_positions = feat.end_position
        cls_index = feat.cls_index
        p_mask = feat.p_mask
        is_impossible = feat.is_impossible

        uid = feat.__dict__['unique_id']

        cached_feat_file = out_ft + f'{prefix}_{uid}'
        torch.save(
          (
            input_ids, attention_masks, token_type_ids, 
            start_positions, end_positions, cls_index,
            p_mask, is_impossible
          ),
          cached_feat_file
        )  

def main():

    tokenizer = AutoTokenizer.from_pretrained(
        configs.tokenizer_name if configs.tokenizer_name else configs.pretrained_model_name_or_path,
        do_lower_case=configs.do_lower_case,
        cache_dir=configs.cache_dir if configs.cache_dir else None,
        use_fast=False,  # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
    )
    load_and_cache_examples(configs.domain_names[0], configs.train_file[0], tokenizer, evaluate=False, output_examples=True)
    load_and_cache_examples(configs.domain_names[1], configs.train_file[1], tokenizer, evaluate=False, output_examples=True)

    path_out = configs.output_dir
    out_ft = path_out+"features_dir/"
    out_pair = path_out+"qid_list_dir/"
    if not os.path.exists(out_ft):
        os.makedirs(out_ft)
    if not os.path.exists(out_pair):
        os.makedirs(out_pair)
        
    feature_to_folder('train_'+configs.domain_names[0], configs.domain_names[0], out_ft, out_pair)
    feature_to_folder('train_'+configs.domain_names[1], configs.domain_names[1], out_ft, out_pair)
        
        
if __name__ == "__main__":
    main()
