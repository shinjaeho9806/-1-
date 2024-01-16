# [1] Skipgram 모델 생성
from torch import nn

class VanillaSkipgram(nn.Module):
    '''
    계층적 소프트 맥스나 네거티브 샘플링 등은 구현하지 않은 기본 형식
    '''
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embedding_dim
        )
        self.linear = nn.Linear(
            in_features = embedding_dim,
            out_features = vocab_size 
        )
    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        output = self.linear(embeddings)
        return output