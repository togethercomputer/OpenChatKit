import os
import re
import torch
import json
from torch.utils.data import IterableDataset, DataLoader
from itertools import cycle, islice
import random
from datasets import Dataset
from datasets import load_dataset, load_from_disk
from comm.comm_utils import *



class StreamDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, seq_length=1024):
        
        self.dataset = dataset
        
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        self.it = None
        self.iter_count = 0
        
    def state_dict(self):
        return {
            'iter_count': self.iter_count,
        }
    
    def load_state_dict(self, state_dict):
        self.iter_count = state_dict['iter_count']
        self.dataset = self.dataset.skip(self.iter_count)
        
    def get_sequence(self):
        
        it = cycle(iter(self.dataset))
        
        while True:

            text_context = '''Possible labels:
1. casual
2. needs caution
3. needs intervention
4. possibly needs caution
5. probably needs caution'''

            while True:
                
                instance = next(it)
                
                text = instance['text']
                text_context += '\n\n' + text
                
                input_ids = self.tokenizer(text_context.strip())['input_ids']
                if len(input_ids) > self.seq_length:
                    break
                
            input_ids = input_ids[:self.seq_length]
            input_ids = torch.tensor(input_ids).long()

            yield {
                'input_ids': input_ids,
            }
            
                
    def get_stream(self):
        return cycle(self.get_sequence())
    
    def __iter__(self):
        if self.it is None:
            self.it = self.get_stream()
        return self.it
    