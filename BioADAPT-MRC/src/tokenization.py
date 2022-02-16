# the tokenization code is inspired from the huggingface example on question-answering task: https://github.com/huggingface/transformers/blob/master/examples/legacy/question-answering/run_squad.py  

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

from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
from transformers.trainer_utils import is_main_process

logger = logging.getLogger(__name__)


def load_and_cache_examples(tokenizer, evaluate=False, output_examples=False):
    if configs.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = configs.output_dir if configs.output_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}_{}_{}".format(
            "dev",
            list(filter(None, configs.original_model_name_or_path.split("/"))).pop(),
            str(configs.max_seq_length),
            "test" if configs.do_test else exit(1),
            configs.dataset_name,
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

        examples = processor.get_dev_examples(configs.data_dir, filename=configs.predict_file)

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


def main():

    tokenizer = AutoTokenizer.from_pretrained(
        configs.tokenizer_name if configs.tokenizer_name else configs.pretrained_model_name_or_path,
        do_lower_case=configs.do_lower_case,
        cache_dir=configs.cache_dir if configs.cache_dir else None,
        use_fast=False,  # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
    )
    load_and_cache_examples(tokenizer, evaluate=True, output_examples=True)

        
if __name__ == "__main__":
    main()
