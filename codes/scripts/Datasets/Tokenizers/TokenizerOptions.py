from enum import Enum


class Emb(int, Enum):
    None
    D64=64
    D128=128

class TokenizerName(str, Enum):
    spacy = 'spacy'
    tiktoken = 'tiktoken'
    debertav3 = 'debertav3'

class TokenizerOptions:
    def __init__(self,
        use_sentiment=True, 
        use_token_embeddings=True, 
        embedding_dim: Emb=Emb.D64,
        tokenizer_name: TokenizerName=TokenizerName.debertav3
    ):
        self.use_sentiment = use_sentiment
        self.use_token_embeddings = use_token_embeddings
        self.embedding_dim = embedding_dim
        self.tokenizer_name = tokenizer_name