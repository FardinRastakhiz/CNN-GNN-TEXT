
from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import torch
from transformers import AutoTokenizer

from . import TokenizerOptions


class Tokenizer(ABC):

    def __init__(self, options: TokenizerOptions):
        self.options = options
        self.tokenizer, self.id_vocab, self.all_vocab_indices, self.all_vocab_str = self._load_tokenizer()
        if self.options.use_token_embeddings:
            self.token_embeddings_dict = self._load_token_embeddings()
        if self.options.use_sentiment:
            self.sentiment_vocab_dict = self._load_sentiments()

    @abstractmethod
    def tokenize(self, text: str):
        pass

    @abstractmethod
    def _load_tokenizer(self) -> tuple[Any, dict[int, str], list[int], list[str]]:
        pass

    @abstractmethod
    def _load_token_embeddings(self) -> dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def _load_sentiments(self)-> dict[str, torch.Tensor]:
        pass
