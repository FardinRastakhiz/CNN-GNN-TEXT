
from codes.scripts.Datasets.ClassificationDatasets.DataManager import DataManager

from . import CnnGnnDataLoader1, CnnGnnDataset1, DatasetOptions, DataManagerOptions
from ..Tokenizers import Tokenizer, TokenizerName, TokenizerOptions, SpacyTokenizer, TiktokenTokenizer, DeBERTaV3Tokenizer
import pandas as pd
from os import path


class AGNewsDataManager(DataManager):

    def __init__(self, options: DataManagerOptions):
        super(AGNewsDataManager, self).__init__(options)

    def _load_dataframes(self):
        data_folder_path = r'data\TextClassification\AGNews'
        test_df = pd.read_csv(
            path.join(data_folder_path, 'test.csv'), header=None)
        test_df['Topic'] = test_df[0] - 1
        test_df['Content'] = test_df[1] + test_df[2]
        train_df = pd.read_csv(
            path.join(data_folder_path, 'train.csv'), header=None)
        train_df['Topic'] = train_df[0] - 1
        train_df['Content'] = train_df[1] + train_df[2]
        return train_df, test_df, pd.concat([train_df, test_df])

    def _setup_classes(self, df: pd.DataFrame):
        classes = ["World", "Sports", "Business", "Sci/Tech"]
        class_list = df.Topic.unique()
        class_id = {classes[i]: i for i in class_list}
        id_class = {i: classes[i] for i in class_list}
        return class_id, id_class

    def _load_tokenizer(self, tokenizer_options: TokenizerName):

        if tokenizer_options.tokenizer_name == TokenizerName.debertav3:
            return DeBERTaV3Tokenizer(tokenizer_options)
        elif tokenizer_options.tokenizer_name == TokenizerName.spacy:
            return SpacyTokenizer(tokenizer_options)
        elif tokenizer_options.tokenizer_name == TokenizerName.tiktoken:
            return TiktokenTokenizer(tokenizer_options)

    def _create_pytorch_datasets(self):
        train_dataset = CnnGnnDataset1(self.train_df.Content.values, self.train_df.Topic.values,
                                       self.options.dataset_options)
        test_dataset = CnnGnnDataset1(self.test_df.Content.values, self.test_df.Topic.values,
                                      self.options.dataset_options)
        #   len(class_id), vocab_dict, token_vocab_dict, t_tokenizer.tokenize)
        return train_dataset, test_dataset

    def _create_pytorch_dataloaders(self):
        train_dataloader = CnnGnnDataLoader1(self.train_dataset,
                                             batch_size=self.options.dataset_options.batch_size, drop_last=True, shuffle=True)
        test_dataloader = CnnGnnDataLoader1(self.test_dataset,
                                            batch_size=self.options.dataset_options.batch_size, drop_last=True, shuffle=False)
        return train_dataloader, test_dataloader

