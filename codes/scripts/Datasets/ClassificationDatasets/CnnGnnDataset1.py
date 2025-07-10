
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch_geometric.data import Data

from codes.scripts.Datasets.ClassificationDatasets.DatasetOptions import DatasetOptions


class CnnGnnDataset1(Dataset):
    
    def __init__(self, X, y, options: DatasetOptions) -> None:
        super().__init__()
        if len(y) % options.batch_size != 0:
            self.shortage = ((len(y) // options.batch_size)+1)*options.batch_size - len(y)
            empty_labels = [i%2 for i in range(self.shortage)]
            empty_strings = [options.tokenizer.id_class[l] for l in empty_labels]
            y = np.concatenate([y, empty_labels])
            X = np.concatenate([X, empty_strings])
        
        y = torch.from_numpy(y)
        self.options = options
        self.y = torch.nn.functional.one_hot(y, num_classes=self.options.num_classes).float()
        self.X = X
        self.max_token_count = 0
        self.all_data = []
        self.token_lengths = []
        self.token_embeddign_ids = []
        
        self.sum_a = 0
        
        for doc in tqdm(self.X):
            g_data = self.content_to_graph(doc, self.options.sampling_equation)
            self.all_data.append(g_data)
        
        self.num_sections = len(y) // self.options.batch_size
        self.x_lengths = np.array([self.all_data[i].character_length for i in range(len(self.all_data))])
        self.x_len_args = np.argsort(self.x_lengths)[::-1]
        
        self.section_ranges = np.linspace(0, len(self.x_len_args), self.num_sections+1)
        self.section_ranges = [(int(self.section_ranges[i-1]), int(self.section_ranges[i])) for i in range(1, len(self.section_ranges))]

        self.position_j = 0
        self.section_i = 0
        self.epoch = 0
        self.each_section_i = np.zeros((self.num_sections, ), dtype=int)
        
        self.sections, self.section_size = self.split_into_k_groups(self.x_len_args, self.x_lengths, self.num_sections)
        
    def __getitem__(self, index):
        index = self.get_section_index()
        return self.all_data[index], self.y[index]
        
    def __len__(self):
        return len(self.y)
    
    def get_section_index(self):
        target_index = self.sections[self.section_i, self.position_j]
        
        self.position_j = (self.position_j + 1) % self.section_size
        if self.position_j == 0:
            self.section_i = (self.section_i + 1) % self.num_sections
            if self.options.shuffle and self.section_i == 0:
                self.sections, self.section_size = self.split_into_k_groups(self.x_len_args, self.x_lengths, self.num_sections)
        return target_index

    def reset_params(self):
        self.section_i = 0
        self.position_j = 0
        self.each_section_i = np.zeros((self.num_sections, ), dtype=int)
        
    def split_into_k_groups(self, len_sorted_args, lengths:np.array, k):
        if self.options.shuffle and self.epoch > 0:
            randomize_sections = np.concatenate([np.random.choice(np.arange(r[0], r[1]), size=r[1]-r[0], replace=False) for r in self.section_ranges])
            len_sorted_args = len_sorted_args[randomize_sections]
        
        nums = lengths[len_sorted_args]
        groups_size = len(len_sorted_args) // k
        
        
        groups = [[] for _ in range(k)]
        group_sums = np.zeros(k, dtype=int)
        group_sizes = np.zeros(k, dtype=int)
        
        for i, num in enumerate(nums):
            candidate_indices = np.where(group_sizes<groups_size)[0]
            min_group_idx = candidate_indices[np.argmin(group_sums[candidate_indices])]
            groups[min_group_idx].append(len_sorted_args[i])
            group_sums[min_group_idx] += num
            group_sizes[min_group_idx] += 1
        self.epoch += 1
        
        groups = np.array(groups)
        group_sums_argsort = np.argsort(group_sums)[::-1]
        groups = groups[group_sums_argsort]
        return np.array(groups), groups_size
        
    def content_to_graph(self, doc, sampling_equation):
        tokens = self.options.tokenizer.tokenize(doc)
        if len(tokens) == 0:
            tokens = ['empty']
                        
        token_lengths = [len(t) for t in tokens]
        tokens.append(self.options.end_of_text_char)
        
        token_lengths.append(len(tokens[-1])-1) # O(1)
        token_lengths = torch.from_numpy(np.array(token_lengths, dtype=np.longlong))+1  # O(n)
        token_embs = [self.options.token_dict[t] if t in self.options.token_dict else torch.zeros((64, ), dtype=torch.float32) for t in tokens] # O(n)
        token_sentiments = [self.options.sentiment_dict[t] if t in self.options.sentiment_dict else (0.0, 0.0) for t in tokens] # O(n)
        token_embs = torch.from_numpy(np.array(token_embs, dtype=np.float32))
        token_sentiments = torch.from_numpy(np.array(token_sentiments, dtype=np.float32))
        doc = ' '.join(tokens)
        characters = torch.from_numpy(np.array([ord(t) if ord(t)<self.options.vocab_size else (self.vocab_size-1) for t in doc], dtype=np.longlong)) # O(n)
        token_positions = torch.arange(len(token_lengths), dtype=torch.long)
        token_indices = torch.repeat_interleave(token_positions, token_lengths) # O(n)
        if self.options.use_sub_sampling:
            token_subsampling_probabilities = sampling_equation(torch.from_numpy(np.array([self.options.token_frequencies[t] if t in self.options.token_frequencies else 1 for t in tokens]))) # O(n)
        num_tokens = len(token_lengths)
        if num_tokens > self.max_token_count:
            self.max_token_count = num_tokens
        g_data = Data(x=characters,
                        token_positions=token_positions,
                        character_length = len(characters),
                        num_tokens = num_tokens,
                        token_indices=token_indices,
                        token_lengths=token_lengths,
                        token_embeddings=token_embs,
                        token_sentiments=token_sentiments,
                        token_subsampling_probabilities=token_subsampling_probabilities)
        return g_data
 
    def caluculate_batch_token_positions(self, num_tokens, character_length, token_indices):
        cumsum_vals = torch.cumsum(num_tokens, dim=0).roll(1)
        cumsum_vals[0] = 0
        additions = torch.repeat_interleave(cumsum_vals, character_length)
        cumulative_token_indices = token_indices + additions
        return cumulative_token_indices       
    