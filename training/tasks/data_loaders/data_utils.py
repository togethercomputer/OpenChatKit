import os
import re
import torch
import json
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from itertools import cycle, islice
import random
from datasets import Dataset
from datasets import load_dataset, load_from_disk
from comm.comm_utils import *


from itertools import islice
from random import randint

SHOW_DATA = int(os.environ.get('SHOW_DATA', 1))


import os
import re
import torch
from torch.utils.data import IterableDataset, DataLoader
from itertools import cycle, islice
import random
from datasets import Dataset
from datasets import load_dataset, load_from_disk
from comm.comm_utils import *


class StreamDataset(IterableDataset):
    default_doc_separator = '\n'
    def __init__(self, data, tokenizer, seq_length=1024, doc_separator=None, cycling=True):
        self.data = data
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.doc_separator = doc_separator or StreamDataset.default_doc_separator
        self.cycling = cycling
        self.it = None
        self.iter_count = 0
        self.buffer_tokens = []
        
    def state_dict(self):
        return {}
    
    def load_state_dict(self, state_dict):
        pass
        
    def get_sequence(self):
        buffer_tokens = self.buffer_tokens
        for x in self.data:
            self.iter_count += 1
            curr_tokens = self.tokenizer(self.doc_separator + x['text'])['input_ids']
            buffer_tokens += curr_tokens
            while len(buffer_tokens) >= self.seq_length:
                tokens = buffer_tokens[:self.seq_length]
                buffer_tokens = buffer_tokens[self.seq_length:]
                input_ids = torch.tensor(tokens)
                self.buffer_tokens = buffer_tokens # update for restore
                yield {
                    'input_ids': input_ids,
                }
                
    def get_stream(self):
        if self.cycling:
            return cycle(self.get_sequence())
        else:
            return self.get_sequence()
    
    def __iter__(self):
        if self.it is None:
            self.it = self.get_stream()
        return self.it


class StreamDatasetList(IterableDataset):
    def __init__(self, task_names, datasets, sample_probs, tokenizer, seq_length=1024, print_sample_every_n=64, post_processor=None):
        
        self.task_names = task_names
        self.datasets = datasets
        self.sample_probs = sample_probs
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.print_sample_every_n = print_sample_every_n
        self.post_processor = post_processor
        
        self.it = None
        
    def state_dict(self):
        return {}
    
    def load_state_dict(self, state_dict):
        pass
        
    def get_sequence(self):
        
        iterators = [cycle(d.get_sequence()) for d in self.datasets]
        prob_ths = np.cumsum([p / sum(self.sample_probs) for p in self.sample_probs])
        
        global_i = 0
        
        while True:
            
            p = random.random()
            
            for task_name, it, th in zip(self.task_names, iterators, prob_ths):
                if p < th:
                    
                    inputs = next(it)
                    
                    if self.post_processor is not None:
                        inputs = self.post_processor(inputs)
                    
                    if SHOW_DATA:
                        if global_i % self.print_sample_every_n == 0:
                            print(p, th)
                            print(f"**{task_name}**:", self.tokenizer.decode(inputs['input_ids']))
                        
                    yield inputs
                    global_i += 1
                    break
                
    def get_stream(self):
        return cycle(self.get_sequence())
    
    def __iter__(self):
        if self.it is None:
            self.it = self.get_stream()
        return self.it
    
    
def name_to_dataset(task, tokenizer, args):
    
    if task != '':
        data = load_dataset("json", data_files=task, split="train", streaming=True).shuffle(buffer_size=100_000, seed=args.seed)
        dataset = StreamDataset(data, tokenizer, args.seq_length)
        
    return dataset

def name_to_dataset_eval(task, tokenizer, args):
    
    if task != '':
        data = load_dataset("json", data_files=task, split="train", streaming=True)
        dataset = StreamDataset(data, tokenizer, args.seq_length, cycling=False)
        
    return dataset

    
def get_train_data_loader(args, tokenizer, num_workers=1, state_dict=None):
    
    task_list = args.task_name.split(',')
    task_names = []
    datasets = []
    probs = []
    
    print('data_utils: parse task_list')
    
    for task in task_list:
        if ':' in task:
            task, prob = task.strip().split(':')
            prob = float(prob)
        else:
            task = task.strip()
            prob = 1.0
            
        dataset = name_to_dataset(task, tokenizer, args)
            
        print('data_utils:', task, prob)
    
        task_names.append(task)
        datasets.append(dataset)
        probs.append(prob)
    
    stream_dataset = StreamDatasetList(
        task_names, datasets, probs,
        tokenizer=tokenizer, seq_length=args.seq_length)
    
    if state_dict is not None:
        stream_dataset.load_state_dict(state_dict)
    
    train_data_loader = torch.utils.data.DataLoader(stream_dataset,
                                                    batch_size=args.batch_size * args.data_group_size,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    pin_memory=True,
                                                    collate_fn=None)
    
    print('data_utils: get train_data_loader')
    
    return train_data_loader


def get_eval_data_loader(args, tokenizer, num_workers=1, state_dict=None):
    
    task_list = args.task_name.split(',')
    task_names = []
    datasets = []
    probs = []
    
    print('data_utils: parse task_list')
    
    evaluation_data = args.evaluation_data
    
    if evaluation_data is None:
        return None
    
    dataset = name_to_dataset_eval(evaluation_data, tokenizer, args)
    
    train_data_loader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    drop_last=True,
                                                    num_workers=num_workers,
                                                    pin_memory=True,
                                                    collate_fn=None)
    
    return train_data_loader

