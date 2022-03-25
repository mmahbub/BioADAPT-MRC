# a significant portion of the training code is adopted from the huggingface example on question-answering task: https://github.com/huggingface/transformers/blob/master/examples/legacy/question-answering/run_squad.py  

import sys
sys.path.append('../')

import glob
import logging
import os
import random
import timeit

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import transformers
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
    get_raw_scores,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
from transformers.trainer_utils import is_main_process

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

import src.configs as configs
from src.bioadapt_mrc_model import bioadapt_mrc_net
from src.data_generator import qa_dataset, qa_collate

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def set_seed(configs):
    random.seed(configs.seed)
    np.random.seed(configs.seed)
    torch.manual_seed(configs.seed)
    if configs.n_gpu > 0:
        torch.cuda.manual_seed_all(configs.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(model, tokenizer):
    """Train the model"""

    set_seed(configs)
    
    if configs.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    configs.train_batch_size = configs.per_gpu_train_batch_size * max(1, configs.n_gpu)

    train_dataset = qa_dataset()
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=configs.train_batch_size,
                                                   shuffle=False,
                                                   num_workers=configs.num_workers,
                                                   drop_last=False,
                                                   collate_fn=qa_collate)

    if configs.max_steps > 0:
        t_total = configs.max_steps
        configs.num_train_epochs = configs.max_steps // (
                len(train_dataloader) // configs.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // configs.gradient_accumulation_steps * configs.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    
    # check the parameters
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": configs.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=configs.learning_rate,
                      eps=configs.adam_epsilon
                     )
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=configs.warmup_steps,
                                                num_training_steps=t_total*configs.lr_multiplier)
    
    # Check if saved optimizer or scheduler states exist
    if (os.path.isfile(os.path.join(configs.pretrained_model_name_or_path, "optimizer.pt")) 
        and os.path.isfile(os.path.join(configs.pretrained_model_name_or_path, "scheduler.pt"))):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(configs.pretrained_model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(configs.pretrained_model_name_or_path, "scheduler.pt")))

    if configs.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=configs.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if configs.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if configs.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[configs.local_rank], output_device=configs.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", configs.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", configs.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        configs.train_batch_size
        * configs.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if configs.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", configs.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    
    # Check if continuing training from a checkpoint
    if os.path.exists(configs.pretrained_model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = configs.pretrained_model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // configs.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (
                    len(train_dataloader) // configs.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(configs.num_train_epochs), desc="Epoch", disable=configs.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(configs)
    
    ite = 0
    patience = configs.patience_threshold
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=configs.local_rank not in [-1, 0])
        
        local_step = 0
        
        for step, batch in enumerate(epoch_iterator):
            local_step += 1
            
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            
            outputs = model(batch)

            encodings, factoid_qa_outputs, aux_qa_outputs, \
            adv_loss, aux_qa_loss, original_qa_loss, loss = outputs

            if configs.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training

            if configs.gradient_accumulation_steps > 1:
                loss = loss / configs.gradient_accumulation_steps

            if configs.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
                
            tr_loss += loss.item()
            
            if (step + 1) % configs.gradient_accumulation_steps == 0:
                if configs.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), configs.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), configs.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if configs.local_rank in [-1, 0] and configs.logging_steps > 0 and global_step % configs.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if configs.local_rank == -1 and configs.evaluate_during_training:
                        results = evaluate(model, tokenizer, None, in_domain=None, out_domain=None, evaluate_all=False, evaluate_domain_0=False)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / configs.logging_steps, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint
                if configs.local_rank in [-1, 0] and configs.save_steps > 0 and global_step % configs.save_steps == 0:
                    
                    output_dir = os.path.join(configs.output_model_dir, "checkpoint-{}".format(global_step))

                    if not os.path.exists(output_dir) and configs.local_rank in [-1, 0]:
                        os.makedirs(output_dir)

                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save.state_dict(), f'{output_dir}/model.pt')
                    tokenizer.save_pretrained(output_dir)

                    torch.save(configs, os.path.join(output_dir, "training_configs.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if configs.max_steps > 0 and global_step > configs.max_steps:
                epoch_iterator.close()
                break
                
        ite+=1
        
        if (ite%10==0):
            if (configs.reverse_layer_lambda < 0.04):
                configs.reverse_layer_lambda = configs.reverse_layer_lambda+configs.lambda_delta

        if configs.max_steps > 0 and global_step > configs.max_steps:
            train_iterator.close()
            break
            
    if configs.local_rank in [-1, 0]:
        tb_writer.close()
          
    return global_step, tr_loss / global_step


def main():

    set_seed(configs)

    if configs.doc_stride >= configs.max_seq_length - configs.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    if (
            os.path.exists(configs.output_dir)
            and os.listdir(configs.output_dir)
            and configs.do_train
            and not configs.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                configs.output_dir
            )
        )

    # Setup distant debugging if needed
    if configs.server_ip and configs.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(configs.server_ip, configs.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if configs.local_rank == -1 or configs.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not configs.no_cuda else "cpu")
        configs.n_gpu = 0 if configs.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(configs.local_rank)
        device = torch.device("cuda", configs.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        configs.n_gpu = 1
    configs.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if configs.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        configs.local_rank,
        device,
        configs.n_gpu,
        bool(configs.local_rank != -1),
        configs.fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(configs.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    
    # Set seed
    set_seed(configs)

    # Load pretrained model and tokenizer
    if configs.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    configs.model_type = configs.model_type.lower()

    tokenizer = AutoTokenizer.from_pretrained(
        configs.tokenizer_name if configs.tokenizer_name else configs.pretrained_model_name_or_path,
        do_lower_case=configs.do_lower_case,
        cache_dir=configs.cache_dir if configs.cache_dir else None,
        use_fast=False,  # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
    )

    if configs.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model = bioadapt_mrc_net()
    
    model.to(configs.device)

    logger.info("Training/evaluation parameters %s", configs)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if configs.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if configs.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")


    # Training
    if configs.do_train:      
        global_step, tr_loss = train(model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if configs.do_train and (configs.local_rank == -1 or torch.distributed.get_rank() == 0):

        if not os.path.exists(configs.output_model_dir):
            os.makedirs(configs.output_model_dir)

        logger.info("Saving model checkpoint to %s", configs.output_model_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, 'module') else model
        
        torch.save(model_to_save.state_dict(), f'{configs.output_model_dir}model.pt')

        tokenizer.save_pretrained(configs.output_model_dir)


if __name__ == "__main__":
    main()
