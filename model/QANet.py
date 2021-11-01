# -*- coding: utf-8 -*-
"""
Main model architecture.
reference: https://github.com/andy840314/QANet-pytorch-
"""
import sys
import os
sys.path.append(os.path.dirname("C://Users//QYS//Desktop//QANet-paddle//model"))
import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from .modules.cnn import DepthwiseSeparableConv
from .modules.attention import MultiHeadAttention
from .modules.position import PositionalEncoding
# from .modules.highway import Highway

# revised two things: head set to 1, d_model set to 96

# print("111",paddle.device.is_compiled_with_cuda())
# device = paddle.device("gpu" if paddle.device.is_compiled_with_cuda() else "cpu")


def mask_logits(target, mask):
    mask = mask.astype(paddle.float32)
    return target * mask + (1 - mask) * (-1e30)  # !!!!!!!!!!!!!!!  do we need * mask after target?


class Initialized_Conv1d(nn.Layer):
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, padding=0, groups=1,
                 relu=False, bias=False):
        super().__init__()
        self.out = nn.Conv1D(
            in_channels, out_channels,
            kernel_size, stride=stride,
            padding=padding, groups=groups, bias_attr=bias)
        if relu is True:
            self.relu = True
            # self.out = nn.Conv1D(
            #     in_channels, out_channels,
            #     kernel_size, stride=stride,
            #     padding=padding, groups=groups, bias_attr=bias, weight_attr=nn.initializer.KaimingNormal())
            self.out.weight = self.create_parameter(
            shape=self.out.weight.shape, default_initializer=nn.initializer.KaimingNormal()
            )

        else:
            self.relu = False
            # self.out = nn.Conv1D(
            #     in_channels, out_channels,
            #     kernel_size, stride=stride,
            #     padding=padding, groups=groups, bias_attr=bias, weight_attr=nn.initializer.XavierUniform())
            self.out.weight = self.create_parameter(
            shape=self.out.weight.shape, default_initializer=nn.initializer.XavierUniform()
            )

    def forward(self, x):
        if self.relu is True:
            return F.relu(self.out(x))
        else:
            return self.out(x)


def PosEncoder(x, min_timescale=1.0, max_timescale=1.0e4):
    # print("111",x.shape)
    x = paddle.transpose(x, perm=[0,2,1])
    length = x.shape[1]
    channels = x.shape[2]
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    return paddle.transpose(x + signal,perm=[0,2,1])


def get_timing_signal(length, channels,
                      min_timescale=1.0, max_timescale=1.0e4):
    position = paddle.arange(length).astype(paddle.float32)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * paddle.exp(
            paddle.arange(num_timescales).astype(paddle.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = paddle.concat([paddle.sin(scaled_time), paddle.cos(scaled_time)], axis = 1)
    m = nn.Pad2D(padding=[0, (channels % 2), 0, 0], mode="constant")
    signal = m(signal)
    signal = paddle.reshape(signal,[1, length, channels])
    return signal


class DepthwiseSeparableConv(nn.Layer):
    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv1D(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k // 2, bias_attr=False)
        self.pointwise_conv = nn.Conv1D(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias_attr=bias)
    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))


class Highway(nn.Layer):
    def __init__(self, layer_num, size):
        super().__init__()
        self.n = layer_num
        self.linear = nn.LayerList([Initialized_Conv1d(size, size, relu=False, bias=True) for _ in range(self.n)])
        self.gate = nn.LayerList([Initialized_Conv1d(size, size, bias=True) for _ in range(self.n)])

    def forward(self, x):
        #x: shape [batch_size, hidden_size, length]
        dropout = 0.1
        for i in range(self.n):
            gate = F.sigmoid(self.gate[i](x))
            nonlinear = self.linear[i](x)
            nonlinear = F.dropout(nonlinear, p=dropout, training=self.training)
            x = gate * nonlinear + (1 - gate) * x
        return x


class SelfAttention(nn.Layer):
    def __init__(self, d_model, num_head, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.dropout = dropout
        self.mem_conv = Initialized_Conv1d(in_channels=d_model, out_channels=d_model*2, kernel_size=1, relu=False, bias=False)
        self.query_conv = Initialized_Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1, relu=False, bias=False)

        bias = paddle.empty([1])
        self.bias = self.create_parameter(
            shape=bias.shape, default_initializer=nn.initializer.Constant(0)
        )

    def forward(self, queries, mask):
        memory = queries

        memory = self.mem_conv(memory)
        query = self.query_conv(queries)
        # print("memory", memory.shape)
        # print("query", query.shape)
        memory = paddle.transpose(memory,perm=[0, 2, 1])
        query = paddle.transpose(query, perm=[0, 2, 1])

        # print("d_model",self.d_model)
        # print(paddle.split(memory, self.d_model, axis=2))
        # for i in paddle.split(memory, memory.shape[2] // self.d_model, axis=2):
        #     print(i.shape)
        # exit(0)
        Q = self.split_last_dim(query, self.num_head)
        K, V = [self.split_last_dim(tensor, self.num_head) for tensor in paddle.split(memory, memory.shape[2] // self.d_model, axis=2)]

        key_depth_per_head = self.d_model // self.num_head
        Q *= key_depth_per_head**-0.5
        x = self.dot_product_attention(Q, K, V, mask = mask)
        return paddle.transpose(self.combine_last_two_dim(paddle.transpose(x,perm = [0,2,1,3])), perm = [0,2,1])

    def dot_product_attention(self, q, k ,v, bias = False, mask = None):
        """dot-product attention.
        Args:
        q: a Tensor with shape [batch, heads, length_q, depth_k]
        k: a Tensor with shape [batch, heads, length_kv, depth_k]
        v: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        is_training: a bool of training
        scope: an optional string
        Returns:
        A Tensor.
        """
        logits = paddle.matmul(q,paddle.transpose(k,perm = [0,1,3,2]))
        if bias:
            logits += self.bias
        if mask is not None:
            shapes = [x  if x != None else -1 for x in list(logits.shape)]
            mask = paddle.reshape(mask, shape=[shapes[0], 1, 1, shapes[-1]])
            logits = mask_logits(logits, mask)
        weights = F.softmax(logits, axis=-1)
        # dropping out the attention links for each of the heads
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        return paddle.matmul(weights, v)

    def split_last_dim(self, x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
        x: a Tensor with shape [..., m]
        n: an integer.
        Returns:
        a Tensor with shape [..., n, m/n]
        """
        old_shape = list(x.shape)
        last = old_shape[-1]

        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        # print("111",old_shape,last,new_shape)
        ret = paddle.reshape(x,shape=new_shape)
        if ret.dim == 3:
            return paddle.transpose(ret, perm=[0, 2, 1])
        else:
            return paddle.transpose(ret, perm=[0, 2, 1, 3])

    def combine_last_two_dim(self, x):
        """Reshape x so that the last two dimension become one.
        Args:
        x: a Tensor with shape [..., a, b]
        Returns:
        a Tensor with shape [..., ab]
        """
        old_shape = list(x.shape)
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = paddle.reshape(x,  shape=new_shape)
        return ret


class Embedding(nn.Layer):
    def __init__(self, wemb_dim, cemb_dim, d_model,
                 dropout_w=0.1, dropout_c=0.05):
        super().__init__()
        self.conv2d = nn.Conv2D(cemb_dim, d_model, kernel_size = (1,5), padding=0, bias_attr=True, weight_attr=paddle.nn.initializer.KaimingNormal())
        self.conv1d = Initialized_Conv1d(wemb_dim + d_model, d_model, bias=False)
        self.high = Highway(2, d_model)
        self.dropout_w = dropout_w
        self.dropout_c = dropout_c

    def forward(self, ch_emb, wd_emb, length):
        ch_emb = ch_emb.transpose(perm=[0, 3, 1, 2])
        ch_emb = F.dropout(ch_emb, p=self.dropout_c, training=self.training)
        ch_emb = self.conv2d(ch_emb)
        ch_emb = F.relu(ch_emb)
        ch_emb = paddle.max(ch_emb, axis=3)

        wd_emb = F.dropout(wd_emb, p=self.dropout_w, training=self.training)
        wd_emb = paddle.transpose(wd_emb, perm=[0,2,1])
        emb = paddle.concat([ch_emb, wd_emb], axis=1)
        emb = self.conv1d(emb)
        emb = self.high(emb)
        return emb


class EncoderBlock(nn.Layer):
    def __init__(self, conv_num, d_model, num_head, k, dropout=0.1):
        super().__init__()
        self.convs = nn.LayerList([DepthwiseSeparableConv(d_model, d_model, k) for _ in range(conv_num)])
        self.self_att = SelfAttention(d_model, num_head, dropout=dropout)
        self.FFN_1 = Initialized_Conv1d(d_model, d_model, relu=True, bias=True)
        self.FFN_2 = Initialized_Conv1d(d_model, d_model, bias=True)
        self.norm_C = nn.LayerList([nn.LayerNorm(d_model) for _ in range(conv_num)])
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.conv_num = conv_num
        self.dropout = dropout

    def forward(self, x, mask, l, blks):
        total_layers = (self.conv_num + 1) * blks
        dropout = self.dropout
        out = PosEncoder(x)
        for i, conv in enumerate(self.convs):
            res = out
            out = self.norm_C[i](out.transpose(perm=[0,2,1])).transpose(perm=[0,2,1])
            if (i) % 2 == 0:
                out = F.dropout(out, p=dropout, training=self.training)
            out = conv(out)
            out = self.layer_dropout(out, res, dropout*float(l)/total_layers)
            l += 1
        res = out
        out = self.norm_1(out.transpose(perm=[0,2,1])).transpose(perm=[0,2,1])
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.self_att(out, mask)
        out = self.layer_dropout(out, res, dropout*float(l)/total_layers)
        l += 1
        res = out

        out = self.norm_2(out.transpose(perm=[0,2,1])).transpose(perm=[0,2,1])
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.FFN_1(out)
        out = self.FFN_2(out)
        out = self.layer_dropout(out, res, dropout*float(l)/total_layers)
        return out

    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            # print("1111",paddle.empty([1]).shape)
            pred = paddle.uniform(shape=paddle.empty([1]).shape, min=0,max=1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual


class CQAttention(nn.Layer):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        w4C = paddle.empty([d_model, 1])
        w4Q = paddle.empty([d_model, 1])
        w4mlu = paddle.empty([1, 1, d_model])
        self.w4C = self.create_parameter(shape=w4C.shape, default_initializer=nn.initializer.XavierUniform())
        self.w4Q = self.create_parameter(shape=w4Q.shape, default_initializer=nn.initializer.XavierUniform())
        self.w4mlu = self.create_parameter(shape=w4mlu.shape, default_initializer=nn.initializer.XavierUniform())

        bias = paddle.empty([1])
        self.bias = self.create_parameter(shape=bias.shape, default_initializer=nn.initializer.Constant(0))
        self.dropout = dropout

    def forward(self, C, Q, Cmask, Qmask):
        C = C.transpose(perm=[0, 2, 1])
        Q = Q.transpose(perm=[0, 2, 1])
        batch_size_c = C.shape[0]
        batch_size, Lc, d_model = C.shape
        batch_size, Lq, d_model = Q.shape
        S = self.trilinear_for_attention(C, Q)
        Cmask = paddle.reshape(Cmask, shape=[batch_size_c, Lc, 1])
        Qmask = paddle.reshape(Qmask, shape=[batch_size_c, 1, Lq])
        S1 = F.softmax(mask_logits(S, Qmask), axis=2)
        S2 = F.softmax(mask_logits(S, Cmask), axis=1)
        A = paddle.bmm(S1, Q)
        B = paddle.bmm(paddle.bmm(S1, S2.transpose(perm=[0, 2, 1])), C)
        out = paddle.concat([C, A, paddle.multiply(C, A), paddle.multiply(C, B)], axis=2)
        return out.transpose(perm=[0, 2, 1])

    def trilinear_for_attention(self, C, Q):
        batch_size, Lc, d_model = C.shape
        batch_size, Lq, d_model = Q.shape
        dropout = self.dropout
        C = F.dropout(C, p=dropout, training=self.training)
        Q = F.dropout(Q, p=dropout, training=self.training)
        subres0 = paddle.matmul(C, self.w4C).expand([-1, -1, Lq])
        subres1 = paddle.matmul(Q, self.w4Q).transpose(perm=[0, 2, 1]).expand([-1, Lc, -1])
        subres2 = paddle.matmul(C * self.w4mlu, Q.transpose([0,2,1]))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res


class Pointer(nn.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.w1 = Initialized_Conv1d(d_model*2, 1)
        self.w2 = Initialized_Conv1d(d_model*2, 1)

    def forward(self, M1, M2, M3, mask):
        X1 = paddle.concat([M1, M2], axis=1)
        X2 = paddle.concat([M1, M3], axis=1)
        Y1 = mask_logits(self.w1(X1).squeeze(), mask)
        Y2 = mask_logits(self.w2(X2).squeeze(), mask)
        return Y1, Y2


class QANet(nn.Layer):
    def __init__(self, word_mat, char_mat,
                 c_max_len, q_max_len, d_model, train_cemb=False, pad=0,
                 dropout=0.1, num_head=1):  # !!! notice: set it to be a config parameter later.
        super().__init__()
        if train_cemb:
            self.char_emb = nn.Embedding(char_mat.shape[0],char_mat.shape[1],sparse=False)
            self.char_emb.weight.set_value(char_mat)
            # self.char_emb = nn.Embedding.from_pretrained(char_mat, freeze=False)
        else:
            self.char_emb = nn.Embedding.from_pretrained(char_mat)
        self.word_emb = nn.Embedding(word_mat.shape[0], word_mat.shape[1], sparse=False)
        self.word_emb.weight.set_value(word_mat)
        self.word_emb.weight.stop_gradient = True
        # self.word_emb = nn.Embedding.from_pretrained(word_mat)
        wemb_dim = word_mat.shape[1]
        cemb_dim = char_mat.shape[1]
        self.emb = Embedding(wemb_dim, cemb_dim, d_model)
        self.num_head = num_head
        self.emb_enc = EncoderBlock(conv_num=4, d_model=d_model, num_head=num_head, k=7, dropout=0.1)
        self.cq_att = CQAttention(d_model=d_model)
        self.cq_resizer = Initialized_Conv1d(d_model * 4, d_model)
        self.model_enc_blks = nn.LayerList([EncoderBlock(conv_num=2, d_model=d_model, num_head=num_head, k=5, dropout=0.1) for _ in range(7)])
        self.out = Pointer(d_model)
        self.PAD = pad
        self.Lc = c_max_len
        self.Lq = q_max_len
        self.dropout = dropout

    def forward(self, Cwid, Ccid, Qwid, Qcid):
        maskC = (paddle.ones_like(Cwid) *
                 self.PAD != Cwid).astype("float32")
        maskQ = (paddle.ones_like(Qwid) *
                 self.PAD != Qwid).astype("float32")
        Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)
        Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)
        C, Q = self.emb(Cc, Cw, self.Lc), self.emb(Qc, Qw, self.Lq)
        Ce = self.emb_enc(C, maskC, 1, 1)
        Qe = self.emb_enc(Q, maskQ, 1, 1)
        X = self.cq_att(Ce, Qe, maskC, maskQ)
        M0 = self.cq_resizer(X)
        M0 = F.dropout(M0, p=self.dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, 7)
        M1 = M0
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, 7)
        M2 = M0
        M0 = F.dropout(M0, p=self.dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, 7)
        M3 = M0
        p1, p2 = self.out(M1, M2, M3, maskC)
        return p1, p2

    def summary(self):
        model_parameters = filter(lambda p: not p.stop_gradient, self.parameters())
        params = sum([np.prod(p.shape) for p in model_parameters])
        print('Trainable parameters:', params)


if __name__ == "__main__":
    paddle.seed(12)
    test_EncoderBlock = False
    test_QANet = True
    test_PosEncoder = False

    if test_EncoderBlock:
        batch_size = 32
        seq_length = 20
        hidden_dim = 96
        x = paddle.rand([batch_size, seq_length, hidden_dim])
        m = EncoderBlock(4, hidden_dim, 8, 7, seq_length)
        y = m(x, mask=None)

    if test_QANet:
        # device and data sizes
        # device = ("gpu" if paddle.device.is_compiled_with_cuda() else "cpu")
        devide = "cpu"
        wemb_vocab_size = 5000
        wemb_dim = 300
        cemb_vocab_size = 94
        cemb_dim = 64
        d_model = 96
        batch_size = 32
        q_max_len = 50
        c_max_len = 400
        char_dim = 16

        # fake embedding
        wv_tensor = paddle.rand([wemb_vocab_size, wemb_dim])
        cv_tensor = paddle.rand([cemb_vocab_size, cemb_dim])

        # fake input
        question_lengths = paddle.to_tensor(np.random.randint(1, q_max_len, (batch_size)),dtype="int64")

        question_wids = paddle.zeros([batch_size, q_max_len]).astype("int64")
        question_cids = paddle.zeros([batch_size, q_max_len, char_dim]).astype("int64")
        context_lengths = paddle.to_tensor(np.random.randint(1, c_max_len, (batch_size)),dtype="int64")
        context_wids = paddle.zeros([batch_size, c_max_len]).astype("int64")
        context_cids = paddle.zeros([batch_size, c_max_len, char_dim]).astype("int64")
        for i in range(batch_size):
            question_wids[i, 0:question_lengths[i]] = paddle.to_tensor(np.random.randint(1, question_lengths[i], (1, wemb_vocab_size)),dtype="int64")
            question_cids[i, 0:question_lengths[i], :] = paddle.to_tensor(np.random.randint(1, cemb_vocab_size, (1, question_lengths[i], char_dim)),dtype="int64")
            context_wids[i, 0:context_lengths[i]] = paddle.to_tensor(np.random.randint(1, wemb_vocab_size, (1, context_lengths[i])),dtype="int64")
            context_cids[i, 0:context_lengths[i], :] = paddle.to_tensor(np.random.randint(1, cemb_vocab_size, (1, context_lengths[i], char_dim)),dtype="int64")

        # test whole QANet
        num_head = 1
        qanet = QANet(wv_tensor, cv_tensor,
                      c_max_len, q_max_len, d_model, train_cemb=False, num_head=num_head)
        p1, p2 = qanet(context_wids, context_cids,
                       question_wids, question_cids)
        print(p1.shape)
        print(p2.shape)

    if test_PosEncoder:
        m = PositionalEncoding(d_model=6, max_len=10, dropout=0)
        input = paddle.randn([3, 10, 6])
        output = m(input)
        print(output)
        output2 = PosEncoder(input)
        print(output2)
