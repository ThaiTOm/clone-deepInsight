import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange

class CombinedMarginLoss(torch.nn.Module):
    def __init__(self,
                 s,
                 m1,
                 m2,
                 m3,
                 interclass_filtering_threshold=0):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold

        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False


    def forward(self, logits, labels):
        index_positive = torch.where(labels != -1)[0]

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty
            logits = tensor_mul * logits

        target_logit = logits[index_positive, labels[index_positive].view(-1)]

        if self.m1 == 1.0 and self.m3 == 0.0:
            with torch.no_grad():
                target_logit.arccos_()
                logits.arccos_()
                final_target_logit = target_logit + self.m2
                logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
                logits.cos_()
            logits = logits * self.s

        elif self.m3 > 0:
            final_target_logit = target_logit - self.m3
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        else:
            raise

        return logits

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=8, emb_size=128):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )

    def forward(self, x):
        x = self.projection(x)
        return x

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = nn.Linear(x.size(-1), units)
        x = F.gelu(x)
        x = nn.Dropout(dropout_rate)(x)
    return x

class PatchEncoder(nn.Module):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = nn.Linear(num_patches, projection_dim)
        self.position_embedding = nn.Embedding(num_patches, projection_dim)

    def forward(self, patch):
        positions = torch.arange(0, self.num_patches).unsqueeze(0)
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

class TransformerEncoderLayer(nn.Module):
    def __init__(self, projection_dim, num_heads, hidden_units, dropout_rate):
        super(TransformerEncoderLayer, self).__init__()
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate

        self.norm1 = nn.LayerNorm(projection_dim)
        self.norm2 = nn.LayerNorm(projection_dim)
        self.multihead_attn = nn.MultiheadAttention(projection_dim, num_heads, dropout=dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.mlp = mlp(hidden_units, dropout_rate)

    def forward(self, x):
        # Layer normalization 1.
        x1 = self.norm1(x)
        # Multi-head attention.
        attention_output, _ = self.multihead_attn(x1, x1, x1)
        # Skip connection 1.
        x2 = x + self.dropout1(attention_output)
        # Layer normalization 2.
        x3 = self.norm2(x2)
        # MLP.
        x4 = self.mlp(x3)
        # Skip connection 2.
        encoded_patches = x2 + self.dropout2(x4)
        return encoded_patches

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, num_patches, projection_dim, num_heads, hidden_units, dropout_rate):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(projection_dim, num_heads, hidden_units, dropout_rate)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        return x

# Assuming mlp function is defined
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = nn.Linear(x.size(-1), units)
        x = F.gelu(x)
        x = nn.Dropout(dropout_rate)(x)
    return x

# Assuming num_layers, num_heads, transformer_units, num_patches, projection_dim, and emb_size are defined
def ViT_backbone():
    input_shape = (144, 144, 3)
    num_patches = 324
    projection_dim = 64
    num_layers = 12
    num_heads = 12
    embed_dim = 256
    num_classes = 2539

    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]

    dropout_rate = 0.5

    mlp_head_units = [
        2048,
        1024,
    ]

    inputs = torch.randn([1, 3, 144, 144])  # Assuming batch size is 1
    patches = PatchEmbedding()(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(num_layers):
        encoded_patches = TransformerEncoderLayer(projection_dim, num_heads, transformer_units, dropout_rate)(encoded_patches)

    representation = nn.LayerNorm(projection_dim)(encoded_patches)
    representation = representation.flatten(start_dim=1)
    representation = nn.Dropout(0.5)(representation)
    representation = nn.Linear(in_features=embed_dim * num_patches, out_features=embed_dim, bias=False)(representation)
    representation = nn.BatchNorm1d(num_features=embed_dim, eps=2e-5)(representation)
    representation = nn.Linear(in_features=embed_dim, out_features=num_classes, bias=False)(representation)
    representation = nn.BatchNorm1d(num_features=num_classes, eps=2e-5)(representation)
    return representation

