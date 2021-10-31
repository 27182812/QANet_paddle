import math
import paddle
from paddle import nn



class PositionalEncoding(nn.Layer):
    """
    Add position information to input tensor.
    :Examples:
        >>> m = PositionalEncoding(d_model=6, max_len=10, dropout=0)
        >>> input = torch.randn(3, 10, 6)
        >>> output = m(input)
    """

    def __init__(self, d_model, dropout=0, max_len=5000):
        """
        :param d_model: same with input hidden size
        :param dropout: dropout rate
        :param max_len: maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = paddle.zeros([max_len, d_model])
        position = paddle.arange(0, max_len).unsqueeze(1)
        div_term = paddle.exp(paddle.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :Input: (batch_num, seq_length, hidden_size)
        :Output: (batch_num, seq_length, hidden_size)
        """
        x = x + paddle.to_tensor(self.pe[:, :x.size(1)], stop_gradient=True)
        return self.dropout(x)
