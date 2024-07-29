import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, embed_num, heads_num, num_layers, seq_length, dropout, num_classes, device):
        super(TransformerModel, self).__init__()
        encoder_layer=nn.TransformerEncoderLayer(d_model=embed_num, nhead=heads_num, dim_feedforward=4*embed_num, dropout=dropout, batch_first=True)
        layer_norm=nn.LayerNorm(normalized_shape=embed_num)
        self.encoder=nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers, norm=layer_norm)
        self.position=nn.Embedding(num_embeddings=seq_length, embedding_dim=embed_num)
        self.dropout=nn.Dropout(p=dropout)
        self.seq_length=seq_length
        self.lin1=nn.Linear(in_features=embed_num, out_features=embed_num*2)
        self.lin2=nn.Linear(in_features=embed_num*2, out_features=num_classes)
        self.relu=nn.ReLU()
        self.batch_norm=nn.BatchNorm1d(num_features=embed_num*2)
        self.device=device


    def forward(self, x):
        N, seq_length,_=x.shape
        positions=torch.arange(0, seq_length).unsqueeze(0).expand(N, seq_length).to(device=self.device)
        x=self.dropout(x+self.position(positions))
        x=self.encoder(x)
        x=torch.max(x, dim=1)[0]
        x=self.dropout(x)
        return self.lin2(self.batch_norm(self.relu(self.lin1(x))))