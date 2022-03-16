import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict
from typing import Tuple


class HMCAN(nn.Module):

    def __init__(self, configs, alpha):
        super(HMCAN, self).__init__()
        self.word_length = configs.max_word_length
        self.alpha = alpha

        self.contextual_transform = TextImage_Transformer(
            configs.contextual_transform, configs.contextual_transform.output_dim)

        self.contextual_transform2 = TextImage_Transformer(
            configs.contextual_transform, configs.contextual_transform.output_dim)


        self.conv = nn.Conv2d(2048, 768, 1)
        self.bn = nn.BatchNorm2d(768)

        self.classifier = nn.Sequential(nn.Linear(768*6, 256),
                                        nn.ReLU(True),
                                        nn.BatchNorm1d(256),
                                        nn.Linear(256, 2)
                                        )



    def forward(self, e, f):
        cap_lengths = len(e)

        e_f_mask = torch.ones(cap_lengths, self.word_length).cuda()
        f_e_mask = torch.ones(cap_lengths, 16).cuda()

        e = torch.squeeze(e, dim=1) # [batch_size, 40, 768]
        e1 = e[:, :self.word_length, :]
        e2 = e[:, self.word_length: self.word_length*2, :]
        e3 = e[:, self.word_length*2:, :]
        # e = self.fc(e) # [batch_size, 40, 64]

        f = F.relu(self.bn(self.conv(f)))  # [batch_size, 768, 4, 4]
        f = f.view(f.shape[0], f.shape[1], -1)  # [batch_size, 768, 16]
        f = f.permute([0, 2, 1])  # [batch_size, 16, 768]

        c1_e1_f = self.contextual_transform(e1, e_f_mask, f)
        c1_f_e1 = self.contextual_transform2(f, f_e_mask, e1)
        a = self.alpha

        c1 = a * c1_e1_f + (1 - a) * c1_f_e1

        c2_e2_f = self.contextual_transform(e2, e_f_mask, f)
        c2_f_e2 = self.contextual_transform2(f, f_e_mask, e2)

        c2 = a * c2_e2_f + (1 - a) * c2_f_e2

        c3_e3_f = self.contextual_transform(e3, e_f_mask, f)
        c3_f_e3 = self.contextual_transform2(f, f_e_mask, e3)

        c3 = a * c3_e3_f + (1 - a) * c3_f_e3

        x = torch.cat((c1, c2, c3), dim=1)
        x = self.classifier(x)


        return x



class LayerNormalization(nn.Module):
    def __init__(self, features_count, epsilon=1e-6):
        super().__init__()
        self.gain = nn.Parameter(
            torch.ones(features_count), requires_grad=True)
        self.bias = nn.Parameter(
            torch.zeros(features_count), requires_grad=True)
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gain * (x - mean) / (std + self.epsilon) + self.bias


class TextImage_Transformer(nn.Module):
    def __init__(self, ct: EasyDict, feature_dim: int):
        super().__init__()

        self.input_norm = LayerNormalization(feature_dim)
        input_dim = feature_dim
        self.embedding = PositionalEncoding(
            input_dim, ct.dropout, max_len=1000)

        self.tf = TransformerEncoder(
            ct.num_layers, input_dim, ct.num_heads, input_dim,
            ct.dropout)

        self.use_context = ct.use_context
        if self.use_context:
            self.tf_context = TransformerEncoder(
                ct.atn_ct_num_layers, input_dim, ct.atn_ct_num_heads,
                input_dim, ct.dropout)

        init_network(self, 0.01)

    def forward(self, features, mask, hidden_state):
        features = self.input_norm(features)
        features = self.embedding(features)
        features = self.tf(features, features, features, mask)
        add_after_pool = None
        if self.use_context:
            ctx = self.tf_context(
                hidden_state, features, features, mask)
            add_after_pool = ctx    # ctx.squeeze(1)

        pooled = torch.mean(features, dim=1)
        add_after_pool = torch.mean(add_after_pool, dim=1)
        if add_after_pool is not None:
            pooled = torch.cat([pooled, add_after_pool], dim=-1)
        return pooled

class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout_prob=0., max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, dim).float()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        dimension = torch.arange(0, dim).float()
        div_term = 10000 ** (2 * dimension / dim)
        pe[:, 0::2] = torch.sin(position / div_term[0::2])
        pe[:, 1::2] = torch.cos(position / div_term[1::2])
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.dim = dim

    def forward(self, x, step=None):
        if step is None:
            x = x + self.pe[:x.size(1), :]
        else:
            x = x + self.pe[:, step]
        x = self.dropout(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, layers_count, d_model, heads_count, d_ff, dropout_prob):
        super().__init__()
        self.d_model = d_model
        assert layers_count > 0
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads_count, d_ff, dropout_prob)
                for _ in range(layers_count)])

    def forward(self, query, key, value, mask):
        batch_size, query_len, embed_dim = query.shape
        batch_size, key_len, embed_dim = key.shape
        mask = (1 - mask.unsqueeze(1).expand(batch_size, query_len, key_len))
        mask = mask == 1
        sources = None
        for encoder_layer in self.encoder_layers:
            sources = encoder_layer(query, key, value, mask)
        return sources


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads_count, d_ff, dropout_prob):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention_layer = Sublayer(
            MultiHeadAttention(heads_count, d_model, dropout_prob), d_model)
        self.pointwise_feedforward_layer = Sublayer(
            PointwiseFeedForwardNetwork(d_ff, d_model, dropout_prob), d_model)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, query, key, value, sources_mask):
        sources = self.self_attention_layer(query, key, value, sources_mask)
        sources = self.dropout(sources)
        sources = self.pointwise_feedforward_layer(sources)
        return sources


class Sublayer(nn.Module):
    def __init__(self, sublayer, d_model):
        super(Sublayer, self).__init__()
        self.sublayer = sublayer
        self.layer_normalization = LayerNormalization(d_model)

    def forward(self, *args):
        x = args[0]
        x = self.sublayer(*args) + x
        return self.layer_normalization(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, heads_count, d_model, dropout_prob):
        super().__init__()
        assert d_model % heads_count == 0,\
            f"model dim {d_model} not divisible by {heads_count} heads"
        self.d_head = d_model // heads_count
        self.heads_count = heads_count
        self.query_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.key_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.value_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.final_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=3)
        self.attention = None

    def forward(self, query, key, value, mask=None):
        batch_size, query_len, d_model = query.size()
        d_head = d_model // self.heads_count
        query_projected = self.query_projection(query)
        key_projected = self.key_projection(key)
        value_projected = self.value_projection(value)
        batch_size, key_len, d_model = key_projected.size()
        batch_size, value_len, d_model = value_projected.size()
        query_heads = query_projected.view(
            batch_size, query_len, self.heads_count, d_head).transpose(1, 2)
        key_heads = key_projected.view(
            batch_size, key_len, self.heads_count, d_head).transpose(1, 2)
        value_heads = value_projected.view(
            batch_size, value_len, self.heads_count, d_head).transpose(1, 2)
        attention_weights = self.scaled_dot_product(
            query_heads, key_heads)
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand_as(attention_weights)
            attention_weights = attention_weights.masked_fill(
                mask_expanded, -1e18)
        attention = self.softmax(attention_weights)
        attention_dropped = self.dropout(attention)
        context_heads = torch.matmul(
            attention_dropped, value_heads)
        context_sequence = context_heads.transpose(1, 2)
        context = context_sequence.reshape(
            batch_size, query_len, d_model)
        final_output = self.final_projection(context)
        return final_output

    def scaled_dot_product(self, query_heads, key_heads):
        key_heads_transposed = key_heads.transpose(2, 3)
        dot_product = torch.matmul(
            query_heads, key_heads_transposed)
        attention_weights = dot_product / np.sqrt(self.d_head)
        return attention_weights


class PointwiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_ff, d_model, dropout_prob):
        super(PointwiseFeedForwardNetwork, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Dropout(dropout_prob),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_prob))

    def forward(self, x):
        return self.feed_forward(x)

def truncated_normal_fill(
        shape: Tuple[int], mean: float = 0, std: float = 1,
        limit: float = 2) -> torch.Tensor:
    num_examples = 8
    tmp = torch.empty(shape + (num_examples,)).normal_()
    valid = (tmp < limit) & (tmp > -limit)
    _, ind = valid.max(-1, keepdim=True)
    return tmp.gather(-1, ind).squeeze(-1).mul_(std).add_(mean)

def init_weight_(w, init_gain=1):

    w.copy_(truncated_normal_fill(w.shape, std=init_gain))


def init_network(net: nn.Module, init_std: float):

    for key, val in net.named_parameters():
        if "weight" in key or "bias" in key:
            init_weight_(val.data, init_std)

