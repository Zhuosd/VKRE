import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GIP(nn.Module):
    def __init__(self, args, data_info, user_fea_index, item_fea_index, item_fea_mask, Graph, spk_emb_list):
        super(GIP, self).__init__()
        self.config = args
        self.datainfo = data_info
        # self.device = device

        self.num_users = self.datainfo['n_users']
        self.num_items = self.datainfo['n_items']
        self.n_user_fea = 5
        self.n_item_fea = self.datainfo['n_item_fea']
        self.user_fea_index = user_fea_index
        self.item_fea_index = item_fea_index
        self.item_fea_mask = item_fea_mask

        self.hidden_size = self.config.recdim
        self.n_layers = self.config.n_layers
        self.keep_prob = self.config.keepprob

        self.Graph = Graph
        self.graph_dropout = self.config.g_drop
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.config.emb_dropout)

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users + self.n_user_fea, embedding_dim=self.hidden_size)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items + self.n_item_fea, embedding_dim=self.hidden_size)
        self.spk_emb = spk_emb_list

        self.initial_weights()

    def initial_weights(self):
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def get_user_emb(self, users):
        user_emb = torch.mean(self.embedding_user(self.user_fea_index[users]), dim=1)
        return (torch.mul(user_emb, self.config.ens_ratio) + torch.mul(self.spk_emb[users],
                                                                           (1 - self.config.ens_ratio)))

    def get_item_emb(self, items):
        if self.config.dataset == 'coat':
            item_fea_emb = self.embedding_item(self.item_fea_index[items])
        elif self.config.dataset == 'movielensmini':
            item_fea_emb = self.embedding_item(self.item_fea_index[items] * self.item_fea_mask[items])            
        elif self.config.dataset == 'movielens1m':
            item_fea_emb = self.embedding_item(self.item_fea_index[items] * self.item_fea_mask[items])

        return torch.mean(item_fea_emb, dim=-2)

    def getRating(self, users):
        all_users, all_items = self.graph_propagate(dr=False)
        user_embeds = all_users[users.long()]
        item_embeds = all_items
        return torch.matmul(user_embeds, item_embeds.t())

    def graph_propagate(self, dr=True):
        """ propagate methods for GIP """
        users_emb = self.get_user_emb(torch.arange(self.num_users))
        items_emb = self.get_item_emb(torch.arange(self.num_items))
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.graph_dropout:
            if dr:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            if self.config.emb_dropout != 0 and dr:
                all_emb = self.dropout(all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        gcn_out = torch.mean(embs, dim=1)
        users, items = torch.split(gcn_out, [self.num_users, self.num_items])
        return users, items

    def get_all_emb(self, users, pos_items, neg_items):
        all_users, all_items = self.graph_propagate()

        user_emb = all_users[users]
        pos_item_emb = all_items[pos_items]
        neg_item_emb = all_items[neg_items]

        user_emb_ego = torch.mean(self.embedding_user(self.user_fea_index[users]), dim=1)
        pos_item_ego = self.get_item_emb(pos_items)
        neg_item_ego = self.get_item_emb(neg_items)

        return user_emb, pos_item_emb, neg_item_emb, user_emb_ego, pos_item_ego, neg_item_ego

