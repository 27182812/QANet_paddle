import math
import copy
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def clones(module, N):
    """Produce N identical layers."""
    return nn.LayerList([copy.deepcopy(module) for _ in range(N)])


class ScaledDotProductAttention(nn.Layer):
    """
    Compute 'Scaled Dot Product Attention'
    """

    def __init__(self, dropout=0.0):
        """
        :param dropout: attention dropout rate
        """
        super().__init__()
        self.dropout = dropout

    def forward(self, query, key, value, mask=None):
        """
        :param query: (batch_num, query_length, d_model)
        :param key: (batch_num, key_length, d_model)
        :param value: (batch_num, key_length, d_model)
        """
        d_k = query.size(-1)
        scores = paddle.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill_(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = F.dropout(p_attn, p=self.dropout)
        return paddle.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Layer):
    """
    Compute 'Multi-Head Attention'
    When we calculate attentions, usually key and value are the same tensor.
    For self-attention, query, key, value are all the same tensor.
    """

    def __init__(self, h, d_model, dropout=0.1):
        """
        :param h: number of heads
        :param d_model: hidden size
        :param dropout: attention dropout rate
        """
        super().__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attention = ScaledDotProductAttention(dropout)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        """
        :param query: (batch_num, query_length, d_model)
        :param key: (batch_num, key_length, d_model)
        :param value: (batch_num, key_length, d_model)
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.shape[0]

        # Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            paddle.transpose(paddle.reshape(l(x),[nbatches, -1, self.h, self.d_k]),perm=[0,2,1])
            for l, x in zip(self.linears, (query, key, value))]
            # l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            # for l, x in zip(self.linears, (query, key, value))]

        # Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(
            query, key, value, mask=mask)

        # "Concat" using a view and apply a final linear.
        x = x.transpose(perm = [0, 2, 1]).reshape(
            [nbatches, -1, self.h * self.d_k])
        return self.linears[-1](x), self.attn


if __name__ == "__main__":
    # Test multi-head attention
    n_head = 8
    d_model = 128
    d_k = d_model // n_head
    d_v = d_model // n_head
    batch_num = 10
    len_q = 20
    len_k = 30
    q = paddle.rand([batch_num, len_q, d_model])
    k = paddle.rand([batch_num, len_k, d_model])
    v = k
    model = MultiHeadAttention(n_head, d_model, dropout=0.1)
    output, attn = model(q, k, v)
    print(output.shape)
    print(attn.shape)
