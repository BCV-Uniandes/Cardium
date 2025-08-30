import torch
import torch.nn as nn

class TabEncoder(nn.Module):
    """
    TabTransformer model for enconding clinical tabular data.
    """
    def __init__(self, num_features, dim_embedding, num_heads, num_layers, dropout=0.3):
        super(TabEncoder, self).__init__()
        self.embedding = nn.Sequential(nn.Linear(1, dim_embedding))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_embedding, 
                                                   nhead=num_heads, 
                                                   batch_first=True)
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.linear = nn.Sequential(nn.Linear(num_features * dim_embedding, dim_embedding),
                                    nn.LayerNorm(dim_embedding),
                                    nn.ReLU())
        
        self.mlp =nn.Sequential(
            nn.Linear(dim_embedding, dim_embedding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_embedding, dim_embedding // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_embedding // 2, dim_embedding // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_embedding // 4, 1)) 

    def forward(self, x):
        batch_size, num_features = x.shape
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.view(batch_size, -1)
        x = self.linear(x)
        x = self.mlp(x)
        return x
