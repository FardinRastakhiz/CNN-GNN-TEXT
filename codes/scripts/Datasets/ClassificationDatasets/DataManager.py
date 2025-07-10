from pandas import DataFrame
from codes.scripts.Datasets.ClassificationDatasets import DataManagerOptions
from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader

from . import DatasetOptions
from ..Tokenizers import Tokenizer, TokenizerOptions


class DataManager(ABC):

    def __init__(self, options: DataManagerOptions):
        self.options = options
        self.train_df, self.test_df, df = self._load_dataframes()
        self.class_id, self.id_class = self._setup_classes(df)
        
        self.tokenizer: Tokenizer = self._load_tokenizer(options.tokenizer_options)
        options.dataset_options = self._create_dataset_options()
        
        self.train_dataset, self.test_dataset = self._create_pytorch_datasets()
        self.train_dataloader, self.test_dataloader = self._create_pytorch_dataloaders()

    def get_train_dataloader(self):
        return self.train_dataloader

    def get_test_dataloader(self):
        return self.test_dataloader

    def get_train_dataset(self):
        return self.train_dataset

    def get_test_dataset(self):
        return self.test_dataset
    
    @abstractmethod
    def _load_dataframes(self) -> tuple[DataFrame, DataFrame, DataFrame]:
        pass
    
    @abstractmethod
    def _setup_classes(self, df: DataFrame) -> tuple[dict[str, int], dict[int, str]]:
        pass
    
    @abstractmethod
    def _load_tokenizer(self, tokenizer_options: TokenizerOptions) -> Tokenizer:
        pass
        
    @abstractmethod
    def _create_pytorch_datasets(self)->tuple[Dataset, Dataset]:
        pass
    
    @abstractmethod
    def _create_pytorch_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        pass
       
    def _create_dataset_options(self):
        return DatasetOptions(num_classes=len(self.class_id),
                              tokenizer=self.tokenizer,
                              token_dict=self.tokenizer.token_embeddings_dict,
                              id_class=self.id_class,
                              class_id=self.class_id,
                              sentiment_dict=self.tokenizer.sentiment_vocab_dict,
                              use_sub_sampling=self.options.use_sub_sampling,
                              shuffle=True,
                              batch_size=self.options.batch_size,
                              char_vocab_size=self.options.char_vocab_size)
    