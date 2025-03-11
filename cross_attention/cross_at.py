import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == self.embed_dim, "Embed dimension must be divisible by number of heads"

        # 定义线性层来产生Query、Key、Value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key_value, key_padding_mask=None):
        """
        :param query: Shape [batch_size, query_seq_len, embed_dim]
        :param key_value: Shape [batch_size, kv_seq_len, embed_dim]
        :param key_padding_mask: Shape [batch_size, kv_seq_len] (optional)
        :return: Output tensor of shape [batch_size, query_seq_len, embed_dim]
        """
        batch_size, query_seq_len, kv_seq_len = query.size(0), query.size(1), key_value.size(1)

        # 将输入投影到不同的空间
        q = self.q_proj(query)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)

        # Reshape to separate heads
        q = q.view(batch_size, query_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 如果提供了padding mask，则应用mask
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)

        # 使用注意力权重对值进行加权求和
        attn_output = torch.matmul(attn_weights, v)

        # 将头重新组合成单一的维度
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, query_seq_len, self.embed_dim)

        # 最后的线性映射
        attn_output = self.out_proj(attn_output)

        return attn_output

# 示例
embed_dim = 4
num_heads = 2
batch_size = 1
query_seq_len = 5
kv_seq_len = 7

query = torch.randn(batch_size, query_seq_len, embed_dim)
key_value = torch.randn(batch_size, kv_seq_len, embed_dim)

cross_attn = CrossAttention(embed_dim, num_heads)
output = cross_attn(query, key_value)

print(output.shape)  # 应该输出 [batch_size, query_seq_len, embed_dim]
