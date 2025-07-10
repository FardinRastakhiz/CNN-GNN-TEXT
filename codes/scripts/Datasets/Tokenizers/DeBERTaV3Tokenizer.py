from . import TokenizerOptions
from . import Tokenizer


import numpy as np
import torch
from transformers import AutoTokenizer


from abc import abstractmethod


class DeBERTaV3Tokenizer(Tokenizer):

    def __init__(self, options: TokenizerOptions):
        super(DeBERTaV3Tokenizer, self).__init__(options)

    @abstractmethod
    def tokenize(self, text: str):
        return self.tokenizer.tokenize(text)

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
        id_vocab = {v: k for k, v in self.tokenizer.vocab.items()}
        all_vocab_indices = list(self.id_vocab.keys())
        all_vocab_str = []
        for i in range(len(self.id_vocab)):
            all_vocab_str.append(self.id_vocab[i])
        return tokenizer, id_vocab, all_vocab_indices, all_vocab_str

    def _load_token_embeddings(self):
        embeddings_path = rf'Data\ReducedEmbeddings\deberta_larg_reduced_embeddings_{self.options.target_embeddings}.npy'
        with open(embeddings_path, 'rb') as f:
            embeddings = np.load(f)
        embeddings = torch.from_numpy(embeddings)
        return dict(zip(self.all_vocab_str, embeddings))

    def _load_sentiments(self):
        sentiment_path = r'Data\ReducedEmbeddings\polarity_debertav3_tokens_gpt_mini_emb.npy'
        with open(sentiment_path, 'rb') as f:
            polarities_subjectivities = np.load(f)
        polarities_subjectivities = torch.from_numpy(polarities_subjectivities)
        sentiment_vocab_dict = dict(
            zip(self.all_vocab_str, polarities_subjectivities))
        sentiment_vocab_dict['<n>'] = torch.tensor([0.0, 0.0])
        sentiment_vocab_dict[''] = torch.tensor([0.0, 0.0])
        return sentiment_vocab_dict
