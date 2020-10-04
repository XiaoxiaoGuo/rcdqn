import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import rnn


class BiDAF(nn.Module):
    def __init__(self, config, word_mat):
        super().__init__()
        self.config = config
        self.word_dim = config.glove_dim
        self.word_emb = nn.Embedding(len(word_mat), len(word_mat[0]),
                                     padding_idx=0)
        self.word_emb.weight.data.copy_(torch.from_numpy(word_mat))
        self.word_emb.weight.requires_grad = False
        self.hidden_dim = config.hidden_dim
        self.keep_prob = config.keep_prob

        self.rnn = EncoderRNN(self.word_dim, self.hidden_dim, 1,
                              concat=True,
                              bidir=True,
                              layernorm='None',
                              return_last=False)
        self.qc_att = BiAttention(self.hidden_dim * 2, 1 - self.keep_prob)
        self.linear_1 = nn.Sequential(
            nn.Linear(self.hidden_dim * 8, self.hidden_dim),
            nn.LeakyReLU())

        self.rnn_2 = EncoderRNN(self.hidden_dim, self.hidden_dim, 1,
                                concat=True,
                                bidir=True,
                                layernorm=config.norm,
                                return_last=False)
        self.self_att = BiAttention(self.hidden_dim * 2, 1 - self.keep_prob)
        self.linear_2 = nn.Sequential(
            nn.Linear(self.hidden_dim * 8, self.hidden_dim),
            nn.LeakyReLU())

        self.rnn_sp = EncoderRNN(self.hidden_dim, self.hidden_dim, 1,
                                 concat=True,
                                 bidir=True,
                                 layernorm=config.norm,
                                 return_last=False)
        self.linear_sp = nn.Linear(self.hidden_dim * 2, 1, bias=False)

    def forward(self, context_ids, ques_ids, context_lens,
                start_mapping, end_mapping):
        bsz = ques_ids.size(0)
        ques_num, ques_len = ques_ids.size(1), ques_ids.size(2)
        cont_len = context_ids.size(1)

        with torch.no_grad():
            # (B x K) x Lq
            ques_mask = (ques_ids > 0).float().view(-1, ques_len)
            context_output = self.word_emb(context_ids)
            ques_output = self.word_emb(ques_ids)

        # (B x K) x Lc x H
        context_output = (self.rnn(context_output, context_lens)
                          .unsqueeze(dim=1)
                          .expand(-1, ques_num, -1, -1)
                          .contiguous()
                          .view(bsz * ques_num, cont_len, -1))
        # (B x K) x Lq x H
        ques_output = (self.rnn(
            ques_output.view(-1, ques_len, self.word_dim))
                       .view(bsz * ques_num, ques_len, -1))

        #
        output = self.qc_att(context_output, ques_output, ques_mask)
        output = self.linear_1(output)

        # (B x K)
        context_lens = (context_lens.unsqueeze(dim=-1)
                        .expand(-1, ques_num)
                        .contiguous()
                        .view(-1))

        # if not self.no_self_att:
        output_t = self.rnn_2(output, context_lens)
        with torch.no_grad():
            # (B x K) x Lc
            context_mask = ((context_ids > 0).float()
                            .unsqueeze(dim=1)
                            .expand(-1, ques_num, -1)
                            .contiguous()
                            .view(-1, cont_len))
        output_t = self.self_att(output_t, output_t, context_mask)
        output_t = self.linear_2(output_t)
        # (B x K) x Lc x H
        output = output + output_t

        # if not self.no_final_gru:
        # (B x K) x Lc x H
        sp_output = self.rnn_sp(output, context_lens)
        # B x Lc x m --> (B x K) x Lc x m
        start_output = torch.bmm(
            start_mapping.view(bsz * ques_num, -1, cont_len),
            sp_output[:, :, self.hidden_dim:])
        end_output = torch.bmm(
            end_mapping.view(bsz * ques_num, -1, cont_len),
            sp_output[:, :, :self.hidden_dim])
        
        sp_output = torch.cat([start_output, end_output], dim=-1)
        sp_output_t = self.linear_sp(sp_output)
        return sp_output_t.view(bsz, ques_num)

    def sample_noise(self):
        return


class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        with torch.no_grad():
            m = (x.data.new(size=(x.size(0), 1, x.size(2)))
                 .bernoulli_(1 - dropout))
            mask = m.div_(1 - dropout)
            mask = mask.expand_as(x)
        return mask * x


class EncoderRNN(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat,
                 bidir, layernorm, return_last):
        super().__init__()
        self.layernorm = (layernorm == 'layer')
        if layernorm:
            self.norm = nn.LayerNorm(input_size)

        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(
                nn.GRU(input_size_, output_size_, 1,
                       bidirectional=bidir, batch_first=True))

        self.rnns = nn.ModuleList(self.rnns)
        self.init_hidden = nn.ParameterList(
            [nn.Parameter(
                torch.zeros(size=(2 if bidir else 1, 1, num_units)),
                requires_grad=True) for _ in range(nlayers)])
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for rnn_layer in self.rnns:
                for name, p in rnn_layer.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(p.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(p.data)
                    elif 'bias' in name:
                        p.data.fill_(0.0)
                    else:
                        p.data.normal_(std=0.1)

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous()

    def forward(self, inputs, input_lengths=None):
        bsz, slen = inputs.size(0), inputs.size(1)
        if self.layernorm:
            inputs = self.norm(inputs)
        output = inputs
        outputs = []
        lens = 0
        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()
        for i in range(self.nlayers):
            hidden = self.get_init(bsz, i)
            # output = self.dropout(output)
            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens,
                                                  batch_first=True,
                                                  enforce_sorted=False)
            output, hidden = self.rnns[i](output, hidden)
            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen:
                    # used for parallel
                    # padding = Variable(output.data.new(1, 1, 1).zero_())
                    padding = torch.zeros(
                        size=(1, 1, 1), dtype=output.type(),
                        device=output.device())
                    output = torch.cat(
                        [output,
                         padding.expand(
                             output.size(0),
                             slen - output.size(1),
                             output.size(2))
                         ], dim=1)
            if self.return_last:
                outputs.append(
                    hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            else:
                outputs.append(output)
        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]


class BiAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = LockedDropout(dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)
        self.dot_scale = nn.Parameter(
            torch.zeros(size=(input_size,)).uniform_(1. / (input_size ** 0.5)),
            requires_grad=True)
        self.init_parameters()

    def init_parameters(self):
        # xavier_uniform_(self.input_linear.weight.data, gain=0.1)
        # xavier_uniform_(self.memory_linear.weight.data, gain=0.1)
        return

    def forward(self, context, memory, mask):
        bsz, input_len = context.size(0), context.size(1)
        memory_len = memory.size(1)
        context = self.dropout(context)
        memory = self.dropout(memory)

        input_dot = self.input_linear(context)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(
            context * self.dot_scale,
            memory.permute(0, 2, 1).contiguous())
        att = input_dot + memory_dot + cross_dot
        att = att - 1e30 * (1 - mask[:, None])

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = (F.softmax(att.max(dim=-1)[0], dim=-1)
                      .view(bsz, 1, input_len))
        output_two = torch.bmm(weight_two, context)
        return torch.cat(
            [context, output_one, context * output_one,
             output_two * output_one],
            dim=-1)
