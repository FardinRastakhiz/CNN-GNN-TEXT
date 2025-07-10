from enum import Enum
from . import DatasetOptions
from ..Tokenizers import Tokenizer, TokenizerOptions



class DataManagerOptions:
    def __init__(
        self,
        target_options: TokenizerOptions,
        dataset_options: DatasetOptions=None,
        batch_size: int = 128,
        char_vocab_size: int = 16384,
        use_sub_sampling: bool = False
    ):
        self.dataset_options = dataset_options
        self.tokenizer_options = target_options
        self.batch_size = batch_size
        self.char_vocab_size = char_vocab_size
        self.use_sub_sampling = use_sub_sampling
        

    
