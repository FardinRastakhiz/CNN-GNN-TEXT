import torch
from torch import nn
import torch.nn.functional as F

class SentimentInjection(nn.Module):
    
    def __init__(self, hidden_dim, embedding_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)        
        self.conv1 = nn.Conv1d(2, embedding_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim+embedding_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(embedding_dim)
        
    def forward(self, x, token_sentiments):
        x1 = F.relu_(self.bn1(self.conv1(token_sentiments.T).T))
        x = F.relu_(self.conv2(torch.cat([x, x1], dim=1).T))
        return x