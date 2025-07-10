 
class SpacyTokenizer(Tokenizer):
    
    def __init__(self, options: TokenizerOptions):
        super(SpacyTokenizer, self).__init__(options)
                
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
        self.id_vocab = {v:k for k,v in self.tokenizer.vocab.items()}
        all_vocab_indices = list(self.id_vocab.keys())
        if self.options.use_token_embeddings:
            self.token_vocab_dict, self.all_vocab_str = self.load_token_embeddings(self.options.target_embeddings)
        if self.options.use_sentiment:
            self.sentiment_vocab_dict = self.load_sentiments()
    
    @abstractmethod
    def tokenize(self):
        pass
    
    def load_token_embeddings(self, embedding_size):
        embeddings_path = rf'Data\ReducedEmbeddings\deberta_larg_reduced_embeddings_{embedding_size}.npy'
        with open(embeddings_path, 'rb') as f:
            embeddings = np.load(f)
        embeddings = torch.from_numpy(embeddings)
        all_vocab_str = []
        for i in range(len(self.id_vocab)):
            all_vocab_str.append(self.id_vocab[i])
            
        token_vocab_dict = dict(zip(all_vocab_str, embeddings))
        return token_vocab_dict, all_vocab_str
    
    def load_sentiments(self):
        sentiment_path = r'Data\ReducedEmbeddings\polarity_debertav3_tokens_gpt_mini_emb.npy'
        with open(sentiment_path, 'rb') as f:
            polarities_subjectivities= np.load(f)
        polarities_subjectivities = torch.from_numpy(polarities_subjectivities)
        sentiment_vocab_dict = dict(zip(self.all_vocab_str, polarities_subjectivities))
        sentiment_vocab_dict['<n>'] = torch.tensor([0.0, 0.0])
        sentiment_vocab_dict[''] = torch.tensor([0.0, 0.0])
        return sentiment_vocab_dict
 