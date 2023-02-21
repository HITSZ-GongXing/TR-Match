import torch
import torch.nn as nn
import torch.nn.init as init
from torch import tensor
import math
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch import LongTensor



class TimeEncoder(nn.Module):
    def __init__(self, expand_dim):
        super(TimeEncoder, self).__init__()

        self.time_dim = expand_dim
        self.basis_freq = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim))).float())

    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]
        ts_cos = torch.cos(map_ts).unsqueeze(-1)
        ts_sin = torch.sin(map_ts).unsqueeze(-1)
        harmonic = torch.cat((ts_cos, ts_sin), 3).view(batch_size, seq_len, -1)
        harmonic = harmonic * torch.sqrt(tensor(1/self.time_dim))

        return harmonic  # self.dense(harmonic)

class TimeEncoder_2(nn.Module):
    def __init__(self, expand_dim):
        super(TimeEncoder_2, self).__init__()

        self.time_dim = expand_dim
        self.basis_freq = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim))).float())

    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]
        ts_cos = torch.cos(map_ts).unsqueeze(-1)
        ts_sin = torch.sin(map_ts).unsqueeze(-1)
        harmonic = torch.cat((ts_cos, ts_sin), 3).view(batch_size, seq_len, -1)
        harmonic = harmonic * torch.sqrt(tensor(1/self.time_dim))

        return harmonic  # self.dense(harmonic)

class TimeEncoder_3(nn.Module):
    def __init__(self, expand_dim):
        super(TimeEncoder_3, self).__init__()

        self.time_dim = expand_dim
        self.basis_freq = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim))).float())

    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]
        ts_cos = torch.cos(map_ts).unsqueeze(-1)
        ts_sin = torch.sin(map_ts).unsqueeze(-1)
        harmonic = torch.cat((ts_cos, ts_sin), 3).view(batch_size, seq_len, -1)
        harmonic = harmonic * torch.sqrt(tensor(1/self.time_dim))

        return harmonic  # self.dense(harmonic)


class NeighborEncoder(nn.Module):
    def __init__(self, few, max_nb, embed_dim, num_symbols, dropout, device):
        super(NeighborEncoder, self).__init__()
        self.few = few
        self.max_nb = max_nb
        self.embed_dim = embed_dim
        self.pad_idx = num_symbols
        self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=self.pad_idx).to(device)
        self.pad_tensor = torch.tensor([self.pad_idx], requires_grad=False).to(device)
        self.dropout = nn.Dropout(dropout)
        self.device = device


        self.Bilinear = nn.Bilinear(embed_dim, embed_dim, 1, bias=False).to(self.device)

        self.head = nn.Linear(embed_dim, embed_dim, bias=False).to(self.device)
        self.neighbor = nn.Linear(embed_dim, embed_dim, bias=False).to(self.device)

        self.time_encoder = TimeEncoder(self.embed_dim).to(self.device)

        init.xavier_normal_(self.Bilinear.weight)
        init.xavier_normal_(self.head.weight)
        init.xavier_normal_(self.neighbor.weight)

    def forward(self, task_rel, support_times, query_times, support_meta, query_meta):
        '''
        :param task_rel: [1]
        :param support_times: [K]
        :param query_times: [q]
        :return:
        '''
        support_left, support_connections_left, support_left_degrees, support_right, support_connections_right, support_right_degrees = support_meta
        query_left, query_connections_left, query_left_degrees, query_right, query_connections_right, query_right_degrees = query_meta

        support_left_degrees = support_left_degrees.unsqueeze(-1)
        support_right_degrees = support_right_degrees.unsqueeze(-1)
        query_left_degrees = query_left_degrees.unsqueeze(-1).unsqueeze(-1)
        query_right_degrees = query_right_degrees.unsqueeze(-1).unsqueeze(-1)

        task_rel_embed = self.dropout(self.symbol_emb(task_rel))
        support_times = support_times.unsqueeze(-1)  # [K, 1]

        support_relations_left = support_connections_left[:, :, 0]  # [K, Max]
        support_entities_left = support_connections_left[:, :, 1]  # [K， Max]
        support_times_left = support_connections_left[:, :, 2]  # [K， Max]

        support_relations_right = support_connections_right[:, :, 0]  # [K， Max]
        support_entities_right = support_connections_right[:, :, 1]  # [K， Max]
        support_times_right = support_connections_right[:, :, 2]  # [K， Max]

        support_times_emb = self.time_encoder(support_times)  # [K, 1, 2d]
        support_times_emb = torch.transpose(support_times_emb, 1, 2)  # [K, 2d, 1]
        support_times_left_emb = self.time_encoder(support_times_left)  # [K, Max, 2d]
        support_time_score_left = torch.matmul(support_times_left_emb, support_times_emb)  # [K, Max, 1]
        support_times_right_emb = self.time_encoder(support_times_right)  # [K, Max, 2d]
        support_time_score_right = torch.matmul(support_times_right_emb, support_times_emb)  # [K, Max, 1]

        # 关系自适应
        support_rel_embeds_left = self.dropout(self.symbol_emb(support_relations_left))  # [K, Max, d]
        support_ent_embeds_left = self.dropout(self.symbol_emb(support_entities_left))  # [K, Max, d]
        rel_embed_support = task_rel_embed.unsqueeze(0).repeat(self.few, self.max_nb, 1)  # [K, Max, d]
        support_relation_score_left = self.Bilinear(support_rel_embeds_left, rel_embed_support)  # [K, Max, 1]

        # 消去Padding_embedding的权重，把0替换成 -np.inf
        support_pad_matrix_left = self.pad_tensor.expand_as(support_relations_left)  # [K, Max]
        support_mask_matrix_left = torch.eq(support_relations_left, support_pad_matrix_left).unsqueeze(-1)  # [K, Max, 1]
        support_score_left = support_time_score_left * support_relation_score_left  # [K, Max, 1]
        support_score_left = support_score_left.masked_fill_(support_mask_matrix_left, -np.inf)  # 把0替换成 -np.inf

        support_att_left = torch.softmax(support_score_left, dim=1)  # [K, Max, 1]
        support_att_left = torch.transpose(support_att_left, 1, 2)  # [K, 1, Max]
        left_support = torch.matmul(support_att_left, support_ent_embeds_left).squeeze(1)  # [K, d]
        support_head_embeds_left = self.dropout(self.symbol_emb(support_left))  # [K, d]
        left_support = torch.relu(self.neighbor(left_support)*support_left_degrees + self.head(support_head_embeds_left))  # [K, d]

        # 关系自适应 right
        support_rel_embeds_right = self.dropout(self.symbol_emb(support_relations_right))  # [K, Max, d]
        support_ent_embeds_right = self.dropout(self.symbol_emb(support_entities_right))  # [K, Max, d]
        support_relation_score_right = self.Bilinear(support_rel_embeds_right, rel_embed_support)  # [qK, Max, 1]

        # 消去Padding_embedding的权重，把0替换成 -np.inf
        support_pad_matrix_right = self.pad_tensor.expand_as(support_relations_right)  # [K, Max]
        support_mask_matrix_right = torch.eq(support_relations_right, support_pad_matrix_right).unsqueeze(-1)  # [K, Max, 1]
        support_score_right = support_time_score_right * support_relation_score_right  # [K, Max, 1]
        support_score_right = support_score_right.masked_fill_(support_mask_matrix_right, -np.inf)  # 把0替换成 -np.inf

        support_att_right = torch.softmax(support_score_right, dim=1)  # [K, Max, 1]
        support_att_right = torch.transpose(support_att_right, 1, 2)  # [K, 1, Max]
        support_ent_embeds_right = support_ent_embeds_right # [K, Max, d]
        right_support = torch.matmul(support_att_right, support_ent_embeds_right).squeeze(1)  # [K, d]
        support_head_embeds_right = self.dropout(self.symbol_emb(support_right)) # [K, d]
        right_support = torch.relu(self.neighbor(right_support)*support_right_degrees + self.head(support_head_embeds_right))

        support_set = torch.cat((left_support, right_support), 1)  # [K, 2d]
        support_set = support_set.unsqueeze(0)  # [1, K, 2d]

        # query
        q = query_times.size(0)
        query_times = query_times.unsqueeze(-1)  # [q, 1]
        query_times_emb = self.time_encoder(query_times)  # [q, 1, 2d]

        query_relations_left = query_connections_left[:, :, 0]  # [q, Max]
        query_entities_left = query_connections_left[:, :, 1]  # [q, Max]
        query_times_left = query_connections_left[:, :, 2]  # [q, Max]

        query_relations_right = query_connections_right[:, :, 0]
        query_entities_right = query_connections_right[:, :, 1]
        query_times_right = query_connections_right[:, :, 2]  # [q, Max]

        query_times_emb_q = torch.transpose(query_times_emb, 1, 2)  # [q, 2d, 1]
        query_time_left_emb = self.time_encoder(query_times_left)  # [q, Max, 2d]
        query_time_score_left = torch.bmm(query_time_left_emb, query_times_emb_q)  # [q, Max, 1]
        query_time_right_emb = self.time_encoder(query_times_right)  # [q, Max, 2d]
        query_time_score_right = torch.bmm(query_time_right_emb, query_times_emb_q)  # [q, Max, 1]

        # 关系自适应
        rel_embed_query = task_rel_embed.unsqueeze(0).repeat(q, self.max_nb, 1)  # [q, Max, d]

        query_rel_embeds_left = self.dropout(self.symbol_emb(query_relations_left))  # [q, Max, d]
        query_ent_embeds_left = self.dropout(self.symbol_emb(query_entities_left))  # [q, Max, d]
        query_relation_score_left = self.Bilinear(query_rel_embeds_left, rel_embed_query)  # [q, Max, 1]
        # 消去Padding_embedding的权重，把0替换成 -np.inf
        query_pad_matrix_left = self.pad_tensor.expand_as(query_relations_left)  # [q, Max]
        query_mask_matrix_left = torch.eq(query_relations_left, query_pad_matrix_left).unsqueeze(-1)  # [q, Max, 1]
        query_score_left = query_time_score_left * query_relation_score_left  # [q, Max, 1]
        query_score_left = query_score_left.masked_fill_(query_mask_matrix_left, -np.inf)  # 把0替换成 -np.inf
        query_att_left = torch.softmax(query_score_left, dim=1)  # [q, Max, 1]
        query_att_left = torch.transpose(query_att_left, 1, 2)  # [q, 1, Max]
        left_query = torch.bmm(query_att_left, query_ent_embeds_left)  # [q, 1, d]
        query_head_embeds_left = self.dropout(self.symbol_emb(query_left)).unsqueeze(1)  # [q, 1, d]
        left_query = torch.relu(self.neighbor(left_query)*query_left_degrees + self.head(query_head_embeds_left))

        query_rel_embeds_right = self.dropout(self.symbol_emb(query_relations_right))  # [q, Max, d]
        query_ent_embeds_right = self.dropout(self.symbol_emb(query_entities_right))  # [q, Max, d]
        query_relation_score_right = self.Bilinear(query_rel_embeds_right, rel_embed_query)  # [q, Max, 1]
        # 消去Padding_embedding的权重，把0替换成 -np.inf
        query_pad_matrix_right = self.pad_tensor.expand_as(query_relations_right)  # [q, Max]
        query_mask_matrix_right = torch.eq(query_relations_right, query_pad_matrix_right).unsqueeze(-1)  # [q, Max, 1]
        query_score_right = query_time_score_right * query_relation_score_right  # [q, Max, 1]
        query_score_right = query_score_right.masked_fill_(query_mask_matrix_right, -np.inf)  # 把0替换成 -np.inf
        query_att_right = torch.softmax(query_score_right, dim=1)  # [q, Max, 1]
        query_att_right = torch.transpose(query_att_right, 1, 2)  # [q, 1, Max]
        right_query = torch.bmm(query_att_right, query_ent_embeds_right)  # [q, 1, d]
        query_head_embeds_right = self.dropout(self.symbol_emb(query_right)).unsqueeze(1)  # [q, 1, d]
        right_query = torch.relu(self.neighbor(right_query)*query_right_degrees + self.head(query_head_embeds_right))

        query_set = torch.cat((left_query, right_query), 2)  # [q, 1, 2d]

        return support_set, query_set

# class NeighborEncoder_Mean(nn.Module):
#     def __init__(self, few, max_nb, embed_dim, num_symbols, dropout, device):
#         super(NeighborEncoder_Mean, self).__init__()
#         self.few = few
#         self.max_nb = max_nb
#         self.embed_dim = embed_dim
#         self.pad_idx = num_symbols
#         self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=self.pad_idx).to(device)
#         self.pad_tensor = torch.tensor([self.pad_idx], requires_grad=False).to(device)
#         self.dropout = nn.Dropout(dropout)
#         self.device = device
#
#         # 关系自适应
#         self.Bilinear = nn.Bilinear(embed_dim, embed_dim, 1, bias=False).to(self.device)
#
#         self.head = nn.Linear(embed_dim, embed_dim, bias=False).to(self.device)
#         self.neighbor = nn.Linear(embed_dim, embed_dim, bias=False).to(self.device)
#
#         self.time_encoder = TimeEncoder(self.embed_dim).to(self.device)
#
#         init.xavier_normal_(self.Bilinear.weight)
#         init.xavier_normal_(self.head.weight)
#         init.xavier_normal_(self.neighbor.weight)
#
#     def forward(self, task_rel, support_times, query_times, support_meta, query_meta):
#         '''
#         :param task_rel: [1]
#         :param support_times: [K]
#         :param query_times: [q]
#         :return:
#         '''
#         support_left, support_connections_left, support_left_degrees, support_right, support_connections_right, support_right_degrees = support_meta
#         query_left, query_connections_left, query_left_degrees, query_right, query_connections_right, query_right_degrees = query_meta
#
#         support_left_degrees = support_left_degrees.unsqueeze(-1)
#         support_right_degrees = support_right_degrees.unsqueeze(-1)
#         query_left_degrees = query_left_degrees.unsqueeze(-1).unsqueeze(-1)
#         query_right_degrees = query_right_degrees.unsqueeze(-1).unsqueeze(-1)
#
#         support_entities_left = support_connections_left[:, :, 1]  # [K， Max]
#
#         support_entities_right = support_connections_right[:, :, 1]  # [K， Max]
#
#
#         support_ent_embeds_left = self.dropout(self.symbol_emb(support_entities_left))  # [K, Max, d]
#         left_support = torch.mean(support_ent_embeds_left, dim=1) # [K, d]
#         support_head_embeds_left = self.dropout(self.symbol_emb(support_left))  # [K, d]
#         left_support = torch.relu(self.neighbor(left_support)*support_left_degrees + self.head(support_head_embeds_left))  # [K, d]
#
#         # 关系自适应 right
#         support_ent_embeds_right = self.dropout(self.symbol_emb(support_entities_right))  # [K, Max, d]
#
#         right_support = torch.mean(support_ent_embeds_right, dim=1)  # [K, d]
#         support_head_embeds_right = self.dropout(self.symbol_emb(support_right)) # [K, d]
#         right_support = torch.relu(self.neighbor(right_support)*support_right_degrees + self.head(support_head_embeds_right))
#
#         support_set = torch.cat((left_support, right_support), 1)  # [K, 2d]
#         support_set = support_set.unsqueeze(0)  # [1, K, 2d]
#
#         # query
#         q = query_times.size(0)
#
#         query_entities_left = query_connections_left[:, :, 1]  # [q, Max]
#
#         query_entities_right = query_connections_right[:, :, 1]
#
#
#         # 关系自适应
#
#         query_ent_embeds_left = self.dropout(self.symbol_emb(query_entities_left))  # [q, Max, d]
#
#         left_query = torch.mean(query_ent_embeds_left, dim=1, keepdim=True)  # [q, 1, d]
#         query_head_embeds_left = self.dropout(self.symbol_emb(query_left)).unsqueeze(1)  # [q, 1, d]
#         left_query = torch.relu(self.neighbor(left_query)*query_left_degrees + self.head(query_head_embeds_left))
#
#         query_ent_embeds_right = self.dropout(self.symbol_emb(query_entities_right))  # [q, Max, d]
#
#         right_query = torch.mean(query_ent_embeds_right, dim=1, keepdim=True)  # [q, 1, d]
#         query_head_embeds_right = self.dropout(self.symbol_emb(query_right)).unsqueeze(1)  # [q, 1, d]
#         right_query = torch.relu(self.neighbor(right_query)*query_right_degrees + self.head(query_head_embeds_right))
#
#         query_set = torch.cat((left_query, right_query), 2)  # [q, 1, 2d]
#
#         return support_set, query_set
#
# class WO_NeighborEncoder(nn.Module):
#     def __init__(self, few, max_nb, embed_dim, num_symbols, dropout, device):
#         super(WO_NeighborEncoder, self).__init__()
#         self.few = few
#         self.max_nb = max_nb
#         self.embed_dim = embed_dim
#         self.pad_idx = num_symbols
#         self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=self.pad_idx).to(device)
#         self.pad_tensor = torch.tensor([self.pad_idx], requires_grad=False).to(device)
#         self.dropout = nn.Dropout(dropout)
#         self.device = device
#
#         # 关系自适应
#         # self.Bilinear = nn.Bilinear(embed_dim, embed_dim, 1, bias=False).to(self.device)
#
#         self.head = nn.Linear(embed_dim, embed_dim, bias=False).to(self.device)
#         # self.neighbor = nn.Linear(embed_dim, embed_dim, bias=False).to(self.device)
#
#         # self.time_encoder = TimeEncoder(self.embed_dim).to(self.device)
#
#         # init.xavier_normal_(self.Bilinear.weight)
#         init.xavier_normal_(self.head.weight)
#         # init.xavier_normal_(self.neighbor.weight)
#
#     def forward(self, task_rel, support_times, query_times, support_meta, query_meta):
#         '''
#         :param task_rel: [1]
#         :param support_times: [K]
#         :param query_times: [q]
#         :return:
#         '''
#         support_left, support_connections_left, support_left_degrees, support_right, support_connections_right, support_right_degrees = support_meta
#         query_left, query_connections_left, query_left_degrees, query_right, query_connections_right, query_right_degrees = query_meta
#
#         support_head_embeds_left = self.dropout(self.symbol_emb(support_left))  # [K, d]
#         left_support = torch.relu(self.head(support_head_embeds_left))  # [K, d]
#
#         support_head_embeds_right = self.dropout(self.symbol_emb(support_right)) # [K, d]
#         right_support = torch.relu(self.head(support_head_embeds_right))
#
#         support_set = torch.cat((left_support, right_support), 1)  # [K, 2d]
#         support_set = support_set.unsqueeze(0)  # [1, K, 2d]
#
#         query_head_embeds_left = self.dropout(self.symbol_emb(query_left)).unsqueeze(1)  # [q, 1, d]
#         left_query = torch.relu(self.head(query_head_embeds_left))
#
#         query_head_embeds_right = self.dropout(self.symbol_emb(query_right)).unsqueeze(1)  # [q, 1, d]
#         right_query = torch.relu(self.head(query_head_embeds_right))
#
#         query_set = torch.cat((left_query, right_query), 2)  # [q, 1, 2d]
#
#         return support_set, query_set

class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size,  dropout_prob, device):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 三个参数矩阵
        self.query = nn.Linear(hidden_size, self.all_head_size).to(device)
        self.key = nn.Linear(hidden_size, self.all_head_size).to(device)
        self.value = nn.Linear(hidden_size, self.all_head_size).to(device)

        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        """
        x : [q, K, 2d]
        shape of x: batch_size * seq_length * hidden_size
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # [q, K, 2, d]
        x = x.view(*new_x_shape)

        # [q, 2, K, d]
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        # shape of hidden_states and mixed_*_layer: batch_size * seq_length * hidden_size
        mixed_query_layer = self.query(hidden_states)  # [q, K, 2d]
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # shape of *_layer: batch_size * num_attention_heads * seq_length * attention_head_size
        query_layer = self.transpose_for_scores(mixed_query_layer)  # [q, 2, K, d]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # shape of attention_scores: batch_size * num_attention_heads * seq_length * seq_length
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [q, 2, K, K]

        attention_scores /= math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)  # [q, 2, K, d]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [q, K, 2, d]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # [q, K, 2d]
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

# class AdaptiveMatcher(nn.Module):
#     def __init__(self, embed_dim, device):
#         super(AdaptiveMatcher, self).__init__()
#         self.embed_dim = embed_dim
#         self.time_encoder = TimeEncoder_3(self.embed_dim).to(device)
#
#     def forward(self, support_set, query_set, support_time, query_time):
#         '''
#         :param support_set: [1, K, 2d]
#         :param query_set: [q, 1, 2d]
#         :param support_time: [K]
#         :param query_time: [q]
#         :return:
#         '''
#         q = query_time.size(0)
#         support_set = support_set.repeat(q, 1, 1)
#         support_time = support_time.unsqueeze(-1)  # [K, 1]
#         support_time_emb = self.time_encoder(support_time).squeeze(1)  # [K, 2d]
#         support_time_emb = support_time_emb.unsqueeze(0).repeat(q, 1, 1)  # [q, K, 2d]
#
#         query_time = query_time.unsqueeze(-1)  # [q, 1]
#         query_time_emb = self.time_encoder(query_time)  # [q, 1, 2d]
#         query_time_emb = torch.transpose(query_time_emb, 1, 2)  # [q, 2d, 1]
#
#         time_score = torch.bmm(support_time_emb, query_time_emb)  # [q, K, 1]
#         att = torch.softmax(time_score, dim=1)  # [q, K, 1]
#         att = torch.transpose(att, 1, 2)  # [q, 1, K]
#         S = torch.bmm(att, support_set)  # [q, 1, 2d]
#         S = torch.transpose(S, 1, 2)  # [q, 1, 2d] to [q, 2d, 1]
#         score = torch.bmm(query_set, S)  # [q, 1, 1]
#         score = score.squeeze(-1).squeeze(-1)
#
#         return score

class QueryEncoder(nn.Module):
    """docstring for QueryEncoder"""
    def __init__(self, input_dim, device, process_step):
        super(QueryEncoder, self).__init__()
        self.input_dim = input_dim
        self.device = device
        self.process_step = process_step
        # self.batch_size = batch_size
        self.process = nn.LSTMCell(input_dim, 2*input_dim).to(device)

    def forward(self, support, query):
        '''
        support: (few, support_dim)
        query: (batch_size, query_dim)
        support_dim = query_dim

        return:
        (batch_size, query_dim)
        '''
        assert support.size()[1] == query.size()[1]


        if self.process_step == 0:
            return query
        batch_size = query.size()[0]
        h_r = Variable(torch.zeros(batch_size, 2*self.input_dim)).to(self.device)
        c = Variable(torch.zeros(batch_size, 2*self.input_dim)).to(self.device)
        for step in range(self.process_step):
            h_r_, c = self.process(query, (h_r, c))
            h = query + h_r_[:,:self.input_dim] # (batch_size, query_dim)
            attn = F.softmax(torch.matmul(h, support.t()), dim=1)
            r = torch.matmul(attn, support) # (batch_size, support_dim)
            h_r = torch.cat((h, r), dim=1)

        return h


class Matcher(nn.Module):
    def __init__(self, few, max_nb, embed_dim, num_symbols, num_attention_heads, dropout, process_step, device):
        super(Matcher, self).__init__()
        self.device = device
        self.neighbor_encoder = NeighborEncoder(few=few, max_nb=max_nb, embed_dim=embed_dim, num_symbols=num_symbols, dropout=dropout, device=device).to(device)
        self.support_aggregator = SelfAttention(num_attention_heads=num_attention_heads, hidden_size=2*embed_dim, dropout_prob=dropout, device=device).to(device)
        self.query_encoder = QueryEncoder(input_dim=2*embed_dim, device=device, process_step=process_step).to(device)
        self.time_encoder = TimeEncoder_2(embed_dim).to(device)

    def forward(self, task_rel, support_time, query_time, support_meta, query_meta):
        support_time = support_time.to(self.device)
        support_set, query_set = self.neighbor_encoder(task_rel, support_time, query_time, support_meta, query_meta)
        support_time_u = support_time.unsqueeze(-1)  # [K, 1]
        support_time_emb = self.time_encoder(support_time_u).squeeze(1)  # [K, 2d]
        support_time_emb = support_time_emb.unsqueeze(0)  # [1, K, 2d]
        support_set = support_set + support_time_emb #[1, K, 2d]
        support_set = self.support_aggregator(support_set)
        support_set = support_set.squeeze(0)  #[K, 2d]

        query_set = query_set.squeeze(1)
        query_time_u = query_time.unsqueeze(-1)  # [q, 1]
        query_time_emb = self.time_encoder(query_time_u).squeeze(1)  # [q, 2d]
        query_set = query_set + query_time_emb
        query_f = self.query_encoder(support_set, query_set)
        support_set = torch.mean(support_set, dim=0, keepdim=True)
        score = torch.matmul(query_f, support_set.t()).squeeze()

        return score































