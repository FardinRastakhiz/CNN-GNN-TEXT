from collections.abc import Callable
from enum import Enum
import pickle
from types import FunctionType
from typing import Literal
import numpy as np
import torch
import torch.nn.functional as F
from ..Tokenizers import Tokenizer


class SubsamplingEquations(str, Enum):
    sigmoid = 'sigmoid'
    linear = 'linear'


class DatasetOptions:
    def __init__(
        self,
        num_classes,
        tokenizer: Tokenizer,
        token_dict,
        id_class:dict,
        class_id:dict=None,
        sentiment_dict=None,
        use_sub_sampling: bool = True,
        token_frequencies: str | dict | None = r'Data\ReducedEmbeddings\term_frequencies.pkl',
        sampling_equation: None | SubsamplingEquations | Callable[..., torch.Tensor] = None,
        shuffle=True,
        batch_size=128,
        char_vocab_size=16384,
        end_of_text_char='',
        sub_sampling_threshold=0.00001
    ):
        self.num_classes = num_classes
        self.token_dict = token_dict
        self.sentiment_dict = sentiment_dict
        self.tokenizer: Tokenizer = tokenizer
        self.use_sub_sampling = use_sub_sampling
        if self.use_sub_sampling:
            self.token_frequencies = self.load_frequencies(token_frequencies) if type(
                token_frequencies) == str else token_frequencies
            self.sub_sampling_threshold = sub_sampling_threshold
            self.total_token_count = np.array(
                list(self.token_frequencies.values())).sum()
            self.sampling_equation = self.get_subsampling_equation(
                sampling_equation)
        self.id_class = id_class
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.char_vocab_size = char_vocab_size
        self.end_of_text_char = end_of_text_char

    def load_frequencies(self, path: str):
        with open(path, 'rb') as f:
            term_frequencies = pickle.load(f)
        return term_frequencies

    def get_subsampling_equation(self, sampling_equation: None | SubsamplingEquations | Callable[..., torch.Tensor] = None):
        if type(sampling_equation) == SubsamplingEquations:
            if sampling_equation == SubsamplingEquations.linear:
                return self.subsampling_equation_linear
            if sampling_equation == SubsamplingEquations.sigmoid:
                return self.subsampling_equation_sigmoid
        elif isinstance(sampling_equation, FunctionType):
            return sampling_equation
        return self.subsampling_equation_sigmoid

    def subsampling_equation_linear(self, x: torch.Tensor):
        f_x = x/self.total_token_count
        x = torch.min(torch.tensor(1), torch.sqrt_(
            self.sub_sampling_threshold/f_x))
        return x

    def subsampling_equation_sigmoid(self, x: torch.Tensor):
        f_x = x/self.total_token_count
        x = 1-0.95*F.sigmoid(0.05*((f_x/self.sub_sampling_threshold)-90))
        return x
