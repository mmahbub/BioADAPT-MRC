# coding=utf-8
# the evaluation code is inspired from the huggingface example on question-answering task: https://github.com/huggingface/transformers/blob/master/examples/legacy/question-answering/run_squad.py  

import sys
sys.path.append('../')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = configs.which_gpu  # specify which GPU(s) to be used if multiple gpus

import glob
import logging
import random
import timeit

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer
)
from transformers.data.metrics.squad_metrics import compute_predictions_logits
from transformers.data.processors.squad import SquadResult

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

import src.configs as configs
from src.bioadapt_mrc_model import bioadapt_mrc_net

logger = logging.getLogger(__name__)

def set_seed(configs):
    random.seed(configs.seed)
    np.random.seed(configs.seed)
    torch.manual_seed(configs.seed)
    if configs.n_gpu > 0:
        torch.cuda.manual_seed_all(configs.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()

def load_all_examples(output_examples):
    # Load data features
    input_dir = configs.output_dir if configs.output_dir else "."
    dataset_name = configs.out_domain_name

    prefix = f'test_{dataset_name}'

    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}_{}_{}".format(
            "dev",
            list(filter(None, configs.original_model_name_or_path.split("/"))).pop(),
            str(configs.max_seq_length),
            "test" if configs.do_test else exit(1),
            str(dataset_name),
            ),
        )

    logger.info("Loading features from cached file %s", cached_features_file)
    features_and_dataset = torch.load(cached_features_file)
    features, dataset, examples = (
        features_and_dataset["features"],
        features_and_dataset["dataset"],
        features_and_dataset["examples"],
        )

    if output_examples:
        return dataset, examples, features, prefix, dataset_name
    return dataset, prefix, dataset_name


def evaluate(model, tokenizer):
    dataset, examples, features, prefix, dataset_name = load_all_examples(output_examples=True)
    if not os.path.exists(configs.output_dir) and configs.local_rank in [-1, 0]:
        os.makedirs(configs.output_dir)

    eval_batch_size = configs.per_gpu_eval_batch_size * max(1, configs.n_gpu)

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    # multi-gpu evaluate
    if configs.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # evaluation
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(configs.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if configs.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
                del inputs["token_type_ids"]

            feature_indices = batch[3]
            encoded_output = model.encoder(inputs)
            outputs = model.factoid_qa_output_generator(encoded_output)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs[:2]]

            start_logits, end_logits = output
            result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(configs.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(configs.output_dir, "nbest_predictions_{}_base.json".format(prefix))
    output_null_log_odds_file = None

    if os.path.isfile(output_nbest_file):
        os.remove(output_nbest_file)
        

    predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            configs.n_best_size,
            configs.max_answer_length,
            configs.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            configs.verbose_logging,
            tokenizer,
        )


def main():

    set_seed(configs)

    if configs.doc_stride >= configs.max_seq_length - configs.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    configs.device = device

    # load pretrained model and tokenizer
    if configs.local_rank not in [-1, 0]:
        # make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    configs.model_type = configs.model_type.lower()

    tokenizer = AutoTokenizer.from_pretrained(
        configs.tokenizer_name if configs.tokenizer_name else configs.pretrained_model_name_or_path,
        do_lower_case=configs.do_lower_case,
        cache_dir=configs.cache_dir if configs.cache_dir else None,
        use_fast=False,  # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
    )

    if configs.local_rank in [-1, 0]:
        logger.info("Loading checkpoint %s for evaluation", configs.trained_model_name)
        checkpoints = [configs.output_model_dir]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = siamese_adv_net()
            
            if configs.USE_TRAINED_MODEL:
                model.load_state_dict(torch.load(f'{checkpoint}{configs.trained_model_name}'))  # , force_download=True)

            model.to(configs.device)
            
            evaluate(model, tokenizer)

            if not os.path.exists(configs.output_model_dir):
                os.makedirs(configs.output_model_dir)

            os.system(f"python3 transform_n2b_factoid.py --nbest_path /net/kdinxidk03/opt/NFS/75y/data/qa/output/nbest_predictions_test_{configs.out_domain_name}_base.json --output_path {configs.outut_dir}")
            os.system(f"java -Xmx10G -cp {configs.java_file_path} {configs.golden_data_folder}{configs.golden_files[0]} {configs.output_dir}BioASQform_BioASQ-answer.json | cut -d' ' -f2,3,4 | sed -e 's/ /,/g' >> {configs.output_dir}sacc_lacc_mrr.txt")
            print('\n')

            with open(f'{configs.output_dir}sacc_lacc_mrr.txt', 'r') as f:
                result = f.readlines()
            result = list(map(float, result_1[-1].replace('\n', '').split(',')))
     
            print('----------RESULTS----------')
            print(result)

if __name__ == "__main__":
    main()
