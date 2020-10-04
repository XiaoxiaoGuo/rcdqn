import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import rnn
from torch.nn.init import xavier_uniform_, kaiming_uniform_, constant_

# from https://github.com/andy840314/QANet-pytorch-/blob/master/models.py
def get_timing_signal(max_length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(max_length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales)-1))
    inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    m = nn.ZeroPad2d((0, (channels % 2), 0, 0))
    signal = m(signal)
    signal = signal.view(max_length, channels)
    return signal


class DepthwiseSeparableConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, k, bias=False):
        super().__init__()
        assert k % 2 == 1
        #
        self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k,
                                        groups=in_ch,
                                        padding=k//2, bias=False)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1,
                                        padding=0, bias=bias)
        # self.conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=k,
        #                                                     padding=k//2, bias=bias)
        # self.depthwise_conv = nn.Linear(in_ch, 2, bias=False)
        # self.pointwise_conv = nn.Linear(2, out_ch, bias=False)
        self.activation = nn.LeakyReLU()
        self.init_parameters(bias)

    def init_parameters(self, bias):
        kaiming_uniform_(self.depthwise_conv.weight.data)
        kaiming_uniform_(self.pointwise_conv.weight.data)
        # if bias:
        #     constant_(self.pointwise_conv.bias.data, 0.0)
        # kaiming_uniform_(self.conv.weight.data)
        # if bias:
        #     constant_(self.conv.bias.data, 0.)

    def forward(self, x):
        return self.activation(
                self.pointwise_conv(
                    self.depthwise_conv(
                        x.permute(0, 2, 1)))).permute(0, 2, 1)
        # return self.activation(
        #     self.pointwise_conv(
        #         self.depthwise_conv(x.permute(0, 2, 1)))).permute(0, 2, 1)
        # return self.activation(
        #     self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        # )
#
#
# class DepthwiseSeparableConv(nn.Module):
#     def __init__(self, in_ch, out_ch, k, bias=True):
#         super().__init__()
#         assert k % 2 == 1
#
#         self.conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=k,
#                               padding=k//2, bias=bias)
#         self.activation = nn.LeakyReLU()
#         self.init_parameters(bias)
#
#     def init_parameters(self, bias):
#         kaiming_uniform_(self.conv.weight.data)
#         if bias:
#             constant_(self.conv.bias.data, 0.)
#
#     def forward(self, x):
#         return self.activation(self.conv(x.permute(0, 2, 1))).permute(0, 2, 1)


class HighwayLayer(nn.Module):
    def __init__(self, hidden_dim, num_layer=1):
        super().__init__()
        self.num_layer = num_layer
        self.nonlinear = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.LeakyReLU()
            )
            for _ in range(self.num_layer)])
        self.gate = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim, bias=True),
                nn.Sigmoid()
            )
            for _ in range(self.num_layer)])
        self.init_parameters()

    def init_parameters(self):
        for l, _ in self.gate:
            constant_(l.bias.data, 0.)

    def forward(self, x):
        for i in range(self.num_layer):
            gate = self.gate[i](x)
            nonlinear = self.nonlinear[i](x)
            x = gate * nonlinear + (1 - gate) * x
        return x


class SelfAttention(nn.Module):
    def __init__(self, emb_dim, num_head=4):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_head = num_head
        self.head_dim = emb_dim // num_head
        assert self.head_dim * self.num_head == self.emb_dim

        self.query = nn.Linear(emb_dim, emb_dim, bias=False)
        self.mem_value = nn.Linear(emb_dim, emb_dim, bias=False)
        self.mem_key = nn.Linear(emb_dim, emb_dim, bias=False)

        self.bias = nn.Parameter(torch.zeros(size=(1,), dtype=torch.float))
        self.init_parameters()

    def init_parameters(self):
        xavier_uniform_(self.query.weight.data, gain=0.01)
        xavier_uniform_(self.mem_key.weight.data, gain=0.01)
        xavier_uniform_(self.mem_value.weight.data, gain=0.01)

    def forward(self, query, mask):
        Q, K, V = self.query(query), self.mem_key(query), self.mem_value(query)
        Q, K, V = self.split_last_dim(Q), self.split_last_dim(K), self.split_last_dim(V)

        Q *= self.head_dim**(-0.5)
        # B x head x Lq x Hh
        x = self.dot_product_attention(Q, K, V, mask=mask)
        # B x Lq x H, no need to mask if it is for self attention
        return self.combine_last_two_dim(x.permute(0, 2, 1, 3))

    def dot_product_attention(self, q, k, v, mask=None):
        # --> B x head x Lq x Lk
        attention = torch.matmul(q, k.permute(0, 1, 3, 2))
        #
        if mask is not None:
            bsz, cont_lens = mask.size(0), mask.size(1)
            mask = mask.view(bsz, 1, 1, cont_lens)
            attention = attention - 1e30 * (1 - mask)
        weights = F.softmax(attention, dim=-1)
        # --> B x head x Lq x Hh
        return torch.matmul(weights, v)

    def split_last_dim(self, x):
        bsz, length = x.size(0), x.size(1)
        x = x.view(bsz, length, self.num_head, self.head_dim)
        # --> head is the second dim
        return x.permute(0, 2, 1, 3)

    def combine_last_two_dim(self, x):
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.contiguous().view(new_shape)
        return ret


class EncoderConvTransformer(nn.Module):
    def __init__(self, num_conv_layers, max_len, in_dim, out_dim, kernel,
                 use_layernorm=True):
        super().__init__()
        self.position_encoding = get_timing_signal(max_len, in_dim)
        self.position_encoding.requires_grad = False

        self.num_conv_layers = num_conv_layers

        if in_dim == out_dim or num_conv_layers == 1:
            self.convs = nn.ModuleList([
                DepthwiseSeparableConvLayer(in_dim, out_dim, kernel)
                for _ in range(num_conv_layers)])
        else:
            first_model = [DepthwiseSeparableConvLayer(in_dim, out_dim, kernel)]
            remaining_models = [
                DepthwiseSeparableConvLayer(out_dim, out_dim, kernel)
                for _ in range(num_conv_layers-1)]
            self.convs = nn.ModuleList(first_model + remaining_models)

        self.self_att = SelfAttention(out_dim)

        self.use_layernorm = use_layernorm
        if use_layernorm:
            if in_dim == out_dim or num_conv_layers == 1:
                self.norm_C = nn.ModuleList([nn.LayerNorm(in_dim)
                                             for _ in range(num_conv_layers)])
            else:
                first_model = [nn.LayerNorm(in_dim)]
                remaining_models = [nn.LayerNorm(out_dim)
                                    for _ in range(num_conv_layers - 1)]
                self.norm_C = nn.ModuleList(first_model + remaining_models)
            self.norm_1 = nn.LayerNorm(out_dim)

        self.init_parameters()

    def init_parameters(self):
        return

    def forward(self, x, mask):
        with torch.no_grad():
            seq_len = x.size(dim=1)
            pos_emb = self.position_encoding[:seq_len, :].unsqueeze(dim=0).to(x.device)
            seq_mask = mask.unsqueeze(dim=-1)
            seq_mask.requires_grad = False

        out = x + pos_emb
        for i, conv in enumerate(self.convs):
            if self.use_layernorm:
                out = self.norm_C[i](out)
            # X: this mask is necessary
            out = out * seq_mask
            out = conv(out)
        # out = out * seq_mask
        out = self.self_att(out, mask)

        if self.use_layernorm:
            out = self.norm_1(out)

        return out


def get_test_inputs(n, k, diff):
    p1 = [torch.Tensor(1, n, k).uniform_(-1, 1),
          torch.Tensor(1, n+diff, k).uniform_(-1, 1),
          torch.Tensor(1, n+2*diff, k).uniform_(-1, 1),
          torch.Tensor(1, n, k).uniform_(-1, 1),
          torch.Tensor(1, n, k).uniform_(-1, 1)]
    p2 = torch.Tensor(5, n+2*diff, k).zero_()
    p2[0, :n, :] = p1[0][0, :, :]
    p2[1, :n+diff, :] = p1[1][0, :, :]
    p2[2, :n+2*diff, :] = p1[2][0, :, :]
    p2[3, :n, :] = p1[3][0, :, :]
    p2[4, :, :] = p2[4, :, :].uniform_(-1, 1)

    p1_len = [torch.ones(size=(1, n)),
              torch.ones(size=(1, n+diff)),
              torch.ones(size=(1, n+2*diff)),
              torch.ones(size=(1, n)),
              torch.ones(size=(1, n))]

    p2_len = torch.zeros(size=(5, n+2*diff))
    p2_len[0, :n] = p1_len[0][0, :]
    p2_len[1, :n + diff] = p1_len[1][0, :]
    p2_len[2, :n + 2 * diff] = p1_len[2][0, :]
    p2_len[3, :n] = p1_len[3][0, :]
    if torch.cuda.is_available():
        p1 = [p.cuda() for p in p1]
        p2 = p2.cuda()
        p1_len = [pl.cuda() for pl in p1_len]
        p2_len = p2_len.cuda()
    return p1, p2, p1_len, p2_len


def check_test_outputs(o1, o2, n, diff):
    error_rate = 1e-7 * 256

    err = (o1[0][0, :, :] - o2[0, :n, :]).abs().sum()
    print('-0 ', err.item())
    if err > error_rate:
        print('#0 is different')

    err = (o1[1][0, :, :] - o2[1, :n+diff, :]).abs().sum()
    print('-1 ', err.item())
    if err > error_rate:
        print('#1 is different')

    err = (o1[2][0, :, :] - o2[2, :n+2*diff, :]).abs().sum()
    print('-2 ', err.item())
    if err > error_rate:
        print('#2 is different')

    err = (o1[3][0, :, :] - o2[3, :n, :]).abs().sum()
    print('-3 ', err.item())
    if err > error_rate:
        print('#3 is different')

    err = (o1[4][0, :, :] - o2[4, :n, :]).abs().sum()
    print('-4 ', err.item())
    if err > error_rate:
        print('#4 is different:')

    return


def test_layer_norm():
    k = 256
    n = 100
    diff = 2

    layer = nn.LayerNorm(k)
    if torch.cuda.is_available():
        layer = layer.cuda()
    p1, p2, _, _ = get_test_inputs(n, k, diff)
    o1 = [layer(x) for x in p1]
    o2 = layer(p2)

    check_test_outputs(o1, o2, n, diff)
    return


# 1e-4 level
def test_highway():
    k = 256
    n = 100
    diff = 2

    layer = HighwayLayer(k)
    if torch.cuda.is_available():
        layer = layer.cuda()
    p1, p2, _, _ = get_test_inputs(n, k, diff)
    o1 = [layer(x) for x in p1]
    o2 = layer(p2)

    check_test_outputs(o1, o2, n, diff)
    return


def test_conv():
    k = 256
    n = 100
    diff = 2

    layer = DepthwiseSeparableConvLayer(k, k, 7)
    if torch.cuda.is_available():
        layer = layer.cuda()
    p1, p2, _, _ = get_test_inputs(n, k, diff)
    o1 = [layer(x) for x in p1]
    o2 = layer(p2)

    for i, p in enumerate(p1):
        print('p1', i, ' # ', p.size())

    print('p2 1 # ', p2.size())

    for i, o in enumerate(o1):
        print('o1', i, ' # ', o.size())

    print('o2 1 # ', o2.size())

    check_test_outputs(o1, o2, n, diff)
    return


# 1e-3 level --> get rid of SelfAttention
def test_self_attention():
    k = 256
    n = 100
    diff = 2

    layer = SelfAttention(k)
    if torch.cuda.is_available():
        layer = layer.cuda()
    p1, p2, p1_len, p2_len = get_test_inputs(n, k, diff)
    o1 = [layer(x, m) for x, m in zip(p1, p1_len)]
    o2 = layer(p2, p2_len)

    check_test_outputs(o1, o2, n, diff)
    return


def test_bidaf():
    k = 256
    n = 100
    diff = 2

    layer = BiAttention(k, 0)
    if torch.cuda.is_available():
        layer = layer.cuda()
    p1, p2, p1_len, p2_len = get_test_inputs(n, k, diff)
    o1 = [layer(x, x, m) for x, m in zip(p1, p1_len)]
    o2 = layer(p2, p2, p2_len)

    check_test_outputs(o1, o2, n, diff)
    return


def test_conv_transformer():
    k = 256
    n = 100
    diff = 2
    # num_conv_layers, max_len, in_dim, out_dim, kernel
    layer = EncoderConvTransformer(num_conv_layers=1, max_len=300, in_dim=k, out_dim=k, kernel=5)
    if torch.cuda.is_available():
        layer = layer.cuda()
    p1, p2, p1_len, p2_len = get_test_inputs(n, k, diff)
    o1 = [layer(x, m) for x, m in zip(p1, p1_len)]
    o2 = layer(p2, p2_len)

    check_test_outputs(o1, o2, n, diff)

    return


def test_rnn():
    # input_size, num_units, nlayers, concat, bidir, dropout, return_last
    # self.word_dim, self.hidden_dim, 1, True, True, 1-self.keep_prob, False
    k = 256
    n = 100
    diff = 2
    # num_conv_layers, max_len, in_dim, out_dim, kernel
    layer = EncoderRNN(k, k, 1, True, True, 0, False)
    if torch.cuda.is_available():
        layer = layer.cuda()
    p1, p2, p1_len, p2_len = get_test_inputs(n, k, diff)
    o1 = [layer(x, m) for x, m in zip(p1, p1_len)]
    o2 = layer(p2, p2_len)

    check_test_outputs(o1, o2, n, diff)

    return
# modular testing
if __name__ == "__main__":
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print('--------- test layer norm -------------')
    test_layer_norm()
    print('--------- test highway -------------')
    test_highway()
    print('--------- test conv -------------')
    test_conv()
    print('--------- test self attention -------------')
    test_self_attention()
    print('--------- test bi-daf attention -------------')
    test_bidaf()
    print('--------- test convTrans  -------------')
    test_conv_transformer()
    print('--------- test rnn  -------------')
    test_rnn()


# similar to GRU based BiDAF model but utilize Transformer and Conv for faster computation
class ConvTransformerBiDAF(nn.Module):
    def __init__(self, config, word_mat):
        super().__init__()
        self.config = config
        self.conv_num = config.conv_num
        self.query_kernel = config.query_kernel
        self.context_kernel = config.context_kernel

        self.word_dim = config.glove_dim
        self.hidden_dim = config.hidden_dim
        self.keep_prob = config.keep_prob

        self.word_emb = nn.Embedding(len(word_mat), len(word_mat[0]), padding_idx=0)
        self.word_emb.weight.data.copy_(torch.from_numpy(word_mat))
        self.word_emb.weight.requires_grad = False

        self.highway = HighwayLayer(num_layer=1, hidden_dim=self.word_dim)

        self.context_encoder = EncoderConvTransformer(self.conv_num,
                                                      self.config.max_obs_seq_len,
                                                      self.word_dim, self.hidden_dim,
                                                      self.context_kernel)
        self.query_encoder = EncoderConvTransformer(self.conv_num,
                                                    self.config.max_template_len,
                                                    self.word_dim, self.hidden_dim,
                                                    self.query_kernel)

        self.qc_att = BiAttention(self.hidden_dim, 1-self.keep_prob)

        self.qc_encoder = EncoderConvTransformer(self.conv_num,
                                                 self.config.max_obs_seq_len,
                                                 self.hidden_dim*4, self.hidden_dim,
                                                 self.context_kernel)

        # self.self_att = BiAttention(self.hidden_dim, 1-self.keep_prob)

        self.output1 = EncoderConvTransformer(self.conv_num,
                                              self.config.max_obs_seq_len,
                                              self.hidden_dim, self.hidden_dim,
                                              self.context_kernel, use_layernorm=False)
        self.output2 = EncoderConvTransformer(self.conv_num,
                                              self.config.max_obs_seq_len,
                                              self.hidden_dim, self.hidden_dim,
                                              self.context_kernel, use_layernorm=False)
        self.obj1_val = nn.Linear(self.hidden_dim, 1)
        self.obj2_val = nn.Linear(self.hidden_dim, 1)
        self.leaky_relu = nn.LeakyReLU()

        self.init_parameters()

    def init_parameters(self):
        xavier_uniform_(self.obj1_val.weight)
        xavier_uniform_(self.obj2_val.weight)
        constant_(self.obj1_val.bias, 0)
        constant_(self.obj2_val.bias, 0)

    def forward(self, context_ids, ques_ids, context_lens,
                start_mapping, end_mapping):

        bsz, ques_num, ques_len, cont_len = ques_ids.size(0), ques_ids.size(1), ques_ids.size(2),\
                                            context_ids.size(1)

        with torch.no_grad():
            # (B x K) x Lq
            ques_mask = (ques_ids > 0).float().view(-1, ques_len)
            context_mask = (context_ids > 0).float()
            context_output = self.word_emb(context_ids)
            ques_output = self.word_emb(ques_ids).view(-1, ques_len, self.word_dim)

        # highway
        # context_output = self.highway(context_output)
        # ques_output = self.highway(ques_output)

        # (B x K) x Lc x H
        context_output = self.context_encoder(context_output, context_mask).unsqueeze(dim=1). \
            expand(-1, ques_num, -1, -1).contiguous().view(bsz * ques_num, cont_len, -1)
        # (B x K) x Lq x H
        ques_output = self.query_encoder(ques_output, ques_mask).\
            view(bsz * ques_num, ques_len, -1)
        #
        output = self.qc_att(context_output, ques_output, ques_mask)

        # (B x K)
        with torch.no_grad():
            context_mask = context_mask.unsqueeze(dim=1).expand(-1, ques_num, -1).\
                contiguous().view(-1, cont_len)

        output = self.qc_encoder(output, context_mask)

        # output_t = self.self_att(output_t, output_t, context_mask)

        # (B x K) x Lc x H,
        # output = output_t + output

        # (B x K) x Lc x H
        obj1_out = self.output1(output, context_mask)

        obj2_out = self.output2(obj1_out, context_mask)
        # (B x K) x Lc x m
        obj1_out = torch.bmm(start_mapping.view(bsz * ques_num, 1, cont_len),
                                    obj1_out)
        obj2_out = torch.bmm(end_mapping.view(bsz * ques_num, 1, cont_len),
                                    obj2_out)

        q_values = self.obj1_val(self.leaky_relu(obj1_out)) + \
                   self.obj2_val(self.leaky_relu(obj2_out))
        return q_values.view(bsz, ques_num)

    def sample_noise(self):
        return
