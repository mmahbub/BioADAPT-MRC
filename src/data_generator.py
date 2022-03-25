import os
import time
import random
import pickle
import torch as t
import numpy as np
from tqdm import tqdm
from torch.utils import data
import gzip
from time import time
import torch
import threading
import src.configs as configs


class qa_dataset(data.Dataset):
    def __init__(self):
        super(qa_dataset, self).__init__()
        
        self.feature_dir = configs.output_dir+"features_dir/"
        self.qid_list_dir = configs.output_dir+"qid_list_dir/"

        self.domain_names = configs.domain_names
        
        self.num_domains = len(self.domain_names)
        
        self.qid_list_dict = {}
        for name in self.domain_names:
            self.qid_list_dict[name] = [qid.replace("\n", "") 
                                        for qid in tqdm(open(self.qid_list_dir + \
                                                             f"{name}_qid_list.txt", "r").readlines())] 
        
        self.data_len = configs.num_samples_per_epoch

    def __getitem__(self, index):
        
        if configs.ALTERNATE_SOURCE_TARGET:
            name_source = self.domain_names[index % self.num_domains]
            ind = index % self.num_domains
        else:
            name_source = self.domain_names[configs.SOURCE_INDEX]
            ind = configs.SOURCE_INDEX
            
        name_target = self.domain_names[random.choice(
                                        list(set(range(self.num_domains)) - \
                                             set([ind])))
                                       ]
        
        qid_list_source = self.qid_list_dict[name_source]
        qid_list_target = self.qid_list_dict[name_target]
       
        data_len_source = len(qid_list_source)
        data_len_target = len(qid_list_target)
        
        qas_0 = qid_list_source[random.randint(0, data_len_source - 1)]
        
        while True:
            qas_1 = qid_list_source[random.randint(0, data_len_source - 1)]
            if qas_0 != qas_1:
                break
                
        qas_2 = qid_list_target[random.randint(0, data_len_target - 1)]

        input_ids0, attention_mask0, token_type_ids0, start_positions0, end_positions0, \
            cls_index0, p_mask0, is_impossible0 = torch.load(self.feature_dir + qas_0)

        input_ids1, attention_mask1, token_type_ids1, start_positions1, end_positions1, \
            cls_index1, p_mask1, is_impossible1 = torch.load(self.feature_dir + qas_1)

        input_ids2, attention_mask2, token_type_ids2, start_positions2, end_positions2, \
            cls_index2, p_mask2, is_impossible2 = torch.load(self.feature_dir + qas_2)

        is_same_domain = [1, 0]

        return torch.tensor([input_ids0], dtype=torch.long), \
               torch.tensor([attention_mask0], dtype=torch.long), \
               torch.tensor([token_type_ids0], dtype=torch.long), \
               torch.tensor([cls_index0], dtype=torch.long), \
               torch.tensor([p_mask0], dtype=torch.long), \
               torch.tensor([is_impossible0], dtype=torch.long), \
               torch.tensor([start_positions0], dtype=torch.long), \
               torch.tensor([end_positions0], dtype=torch.long), \
               torch.tensor([input_ids1], dtype=torch.long), \
               torch.tensor([attention_mask1], dtype=torch.long), \
               torch.tensor([token_type_ids1], dtype=torch.long), \
               torch.tensor([cls_index1], dtype=torch.long), \
               torch.tensor([p_mask1], dtype=torch.long), \
               torch.tensor([is_impossible1], dtype=torch.long), \
               torch.tensor([start_positions1], dtype=torch.long), \
               torch.tensor([end_positions1], dtype=torch.long), \
               torch.tensor([input_ids2], dtype=torch.long), \
               torch.tensor([attention_mask2], dtype=torch.long), \
               torch.tensor([token_type_ids2], dtype=torch.long), \
               torch.tensor([cls_index2], dtype=torch.long), \
               torch.tensor([p_mask2], dtype=torch.long), \
               torch.tensor([is_impossible2], dtype=torch.long), \
               torch.tensor([start_positions2], dtype=torch.long), \
               torch.tensor([end_positions2], dtype=torch.long)

    def __len__(self):
        return self.data_len


def qa_collate(samples):
    input_ids0_batch, attention_mask0_batch, token_type_ids0_batch, \
    cls_index0_batch, p_mask0_batch, is_impossible0_batch, \
    start_positions0_batch, end_positions0_batch, \
    input_ids1_batch, attention_mask1_batch, token_type_ids1_batch, \
    cls_index1_batch, p_mask1_batch, is_impossible1_batch, \
    start_positions1_batch, end_positions1_batch, \
    input_ids2_batch, attention_mask2_batch, token_type_ids2_batch, \
    cls_index2_batch, p_mask2_batch, is_impossible2_batch, \
    start_positions2_batch, end_positions2_batch = map(list, zip(*samples))

    input_ids0_batch = torch.cat(input_ids0_batch)
    attention_mask0_batch = torch.cat(attention_mask0_batch)
    token_type_ids0_batch = torch.cat(token_type_ids0_batch)
    start_positions0_batch = torch.cat(start_positions0_batch)
    end_positions0_batch = torch.cat(end_positions0_batch)

    input_ids1_batch = torch.cat(input_ids1_batch)
    attention_mask1_batch = torch.cat(attention_mask1_batch)
    token_type_ids1_batch = torch.cat(token_type_ids1_batch)
    start_positions1_batch = torch.cat(start_positions1_batch)
    end_positions1_batch = torch.cat(end_positions1_batch)

    input_ids2_batch = torch.cat(input_ids2_batch)
    attention_mask2_batch = torch.cat(attention_mask2_batch)
    token_type_ids2_batch = torch.cat(token_type_ids2_batch)
    start_positions2_batch = torch.cat(start_positions2_batch)
    end_positions2_batch = torch.cat(end_positions2_batch)

    question_context_features = [
                                 {'input_ids': input_ids0_batch, 
                                  'attention_mask': attention_mask0_batch,
                                  'token_type_ids': token_type_ids0_batch
                                 },
                                {'input_ids': input_ids1_batch,
                                 'attention_mask': attention_mask1_batch,
                                 'token_type_ids': token_type_ids1_batch
                                },
                                {'input_ids': input_ids2_batch,
                                 'attention_mask': attention_mask2_batch,
                                 'token_type_ids': token_type_ids2_batch
                                }
                                ]

    if configs.model_type in {"xlm", "roberta", "distilbert", "camembert", "bart", "longformer"}:
        del question_context_features[0]['token_type_ids']
        del question_context_features[1]['token_type_ids']
        del question_context_features[2]['token_type_ids']

    start_positions = [start_positions0_batch, start_positions1_batch, start_positions2_batch]
    end_positions = [end_positions0_batch, end_positions1_batch, end_positions2_batch]

    return question_context_features, start_positions, end_positions
