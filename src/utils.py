import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp
import os
import pickle
import shutil
import json
from collections import Counter

def init_seed(seed=2023):
    """ init random seed for random functions in numpy, torch, cuda and cudnn
    :param seed: random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# ====================Loss=================================
class CosineContrastiveLoss(nn.Module):
    def __init__(self, margin=0, negative_weight=None):
        """
        :param margin: float, margin in CosineContrastiveLoss
        :param num_negs: int, number of negative samples
        :param negative_weight:, float, the weight set to the negative samples. When negative_weight=None, it
            equals to num_negs
        """
        super(CosineContrastiveLoss, self).__init__()
        self._margin = margin
        self._negative_weight = negative_weight

    def forward(self, y_pred, y_true=0):
        """
        :param y_pred: prdicted values of shape (batch_size, 1 + num_negs)
        :param y_true: true labels of shape (batch_size, 1 + num_negs)
        """
        pos_logits = y_pred[:, 0]
        pos_loss = torch.relu(1 - pos_logits)
        neg_logits = y_pred[:, 1:]
        neg_loss = torch.relu(neg_logits - self._margin)
        if self._negative_weight:
            loss = pos_loss + neg_loss.mean(dim=-1) * self._negative_weight
        else:
            loss = pos_loss + neg_loss.sum(dim=-1)
        return loss.mean()

def cosine_sim(user_emb, item_emb):
    batch_size, num_dim = user_emb.shape
    if item_emb.dim() == 3:
        user_emb_expanded = user_emb.unsqueeze(1).expand(-1, item_emb.shape[1], -1)
        user_emb_flat = user_emb_expanded.reshape(-1, num_dim)
        item_emb_flat = item_emb.reshape(-1, num_dim)
        cosim = F.cosine_similarity(user_emb_flat, item_emb_flat, dim=1)
        cosine_pred = cosim.view(batch_size, -1)
    else:
        cosim = F.cosine_similarity(user_emb, item_emb)
        cosine_pred = cosim.view(-1, 1)
    return cosine_pred

class EarlyStopping():
    def __init__(self, patience=100, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.score_max = -np.Inf
        self.delta = delta
    def __call__(self, score, model, args):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, args)
        elif score < self.best_score+self.delta:
            self.counter+=1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter>=self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model, args)
            self.counter = 0
    def save_checkpoint(self, score, model, args):
        if self.verbose:
            print(
                f'Validation score increased ({self.score_max:.6f} --> {score:.6f}).  Saving model ...')
        torch.save(model, f'./checkpoints/{args.dataset}.pt')
        self.score_max = score

# ====================Data=================================
# =========================================================
def get_data_info(dataset, device):
    data_info = {}

    if dataset == 'coat':
        with open(f'../data/coat/info.txt', 'r') as f:
            data_info['n_users'], data_info['n_items'] = [int(n) for n in f.readline().strip('\n').split(' ')]
        item_feature = np.genfromtxt('../data/coat/item_features.ascii', dtype=None)
        data_info['n_item_fea'] = len(item_feature[0]) - 2
        data_info['n_item_fea_type'] = len(np.where(item_feature[0] == 1)[-1][: -1])

        item_feature = np.delete(item_feature, [-1, -2], axis=1)
        index = np.where(item_feature == 1)[-1]
        index = np.reshape(index, (-1, data_info['n_item_fea_type']))
        item_fea_index = np.insert(index, 0, np.arange(0, index.shape[0]), axis=1)
        item_fea_index[:, -3:] += data_info['n_items']

        item_fea_index = torch.tensor(item_fea_index).to(device)

        return data_info, item_fea_index

    elif dataset == 'movielensmini':
        with open(f'../data/movielensmini/info.txt', 'r') as f:
            data_info['n_users'], data_info['n_items'] = [int(n) for n in f.readline().strip('\n').split(' ')]
        item_list = np.genfromtxt('../data/movielensmini/item_list.txt', dtype=None)
        df = pd.read_csv(f'../data/movielensmini/ml-1m.csv')
        df = df.drop(columns=['user_id', 'rating', 'timestamp', 'movie', 'director', 'actors', 'title'])
        df.drop_duplicates(keep='first', inplace=True)
        df = df.sort_values('movie_id')

        one_hot_country = pd.get_dummies(df['country'], prefix='country', dtype=int)
        one_hot_country = one_hot_country.values

        df = pd.concat([df['movie_id'], df['genres']], axis=1)

        num_country = len(one_hot_country[0])
        item_fea_index = []
        uniqueGenre = []
        for (index, row), country in zip(df.iterrows(), one_hot_country):
            if row['movie_id'] not in item_list:
                continue
            country_index = np.where(country == 1)[0]

            genre_index_list = []
            sep_genres = row['genres'].split('|')
            for genre in sep_genres:
                if genre not in uniqueGenre:
                    uniqueGenre.append(genre)
                genre_index_list.append(uniqueGenre.index(genre) + data_info['n_items'] + num_country)
            genre_index_list.sort(key=lambda x: int(x))

            movie_index = int(np.where(item_list == row['movie_id'])[0][0])
            index_list = [movie_index, int(country_index) + data_info['n_items']]
            index_list.extend(genre_index_list)

            item_fea_index.append(torch.tensor(index_list))

        padding_value = 100000
        item_fea_index = torch.nn.utils.rnn.pad_sequence(item_fea_index, batch_first=True, padding_value=padding_value)
        mask = np.array(item_fea_index != padding_value).astype(int)
        mask = torch.tensor(mask).to(device)
        item_fea_index = item_fea_index.to(device)
        data_info['n_item_fea'] = num_country + len(uniqueGenre)

        return data_info, item_fea_index, mask

    elif dataset == 'movielens1m':
        with open(f'../data/movielens1m/info.txt', 'r') as f:
            data_info['n_users'], data_info['n_items'] = [int(n) for n in f.readline().strip('\n').split(' ')]
        item_list = np.genfromtxt('../data/movielens1m/item_list.txt', dtype=None)
        df = pd.read_csv(f'../data/movielens1m/ml-1m.csv')
        df = df.drop(columns=['user_id', 'rating', 'timestamp', 'movie', 'director', 'actors', 'title'])
        df.drop_duplicates(keep='first', inplace=True)
        df = df.sort_values('movie_id')

        one_hot_country = pd.get_dummies(df['country'], prefix='country', dtype=int)
        one_hot_country = one_hot_country.values

        df = pd.concat([df['movie_id'], df['genres']], axis=1)

        num_country = len(one_hot_country[0])
        item_fea_index = []
        uniqueGenre = []
        for (index, row), country in zip(df.iterrows(), one_hot_country):
            if row['movie_id'] not in item_list:
                continue
            country_index = np.where(country == 1)[0]

            genre_index_list = []
            sep_genres = row['genres'].split('|')
            for genre in sep_genres:
                if genre not in uniqueGenre:
                    uniqueGenre.append(genre)
                genre_index_list.append(uniqueGenre.index(genre) + data_info['n_items'] + num_country)
            genre_index_list.sort(key=lambda x: int(x))

            movie_index = int(np.where(item_list == row['movie_id'])[0][0])
            index_list = [movie_index, int(country_index) + data_info['n_items']]
            index_list.extend(genre_index_list)

            item_fea_index.append(torch.tensor(index_list))

        padding_value = 100000
        item_fea_index = torch.nn.utils.rnn.pad_sequence(item_fea_index, batch_first=True, padding_value=padding_value)
        mask = np.array(item_fea_index != padding_value).astype(int)
        mask = torch.tensor(mask).to(device)
        item_fea_index = item_fea_index.to(device)
        data_info['n_item_fea'] = num_country + len(uniqueGenre)

        return data_info, item_fea_index, mask

def get_user_fea_index(dataset, data_info, device):
    with open(f'./user_pred_labels_{dataset}.json', 'r') as f:
        user_pred_labels = json.load(f)

    user_fea_index = []
    for key, value in user_pred_labels.items():
        likely_gender = Counter(np.array(value)[:, 0]).most_common(1)[0][0]
        likely_age = Counter(np.array(value)[:, 1]).most_common(1)[0][0]

        user_fea_index.append([int(key), likely_gender+data_info['n_users'], likely_age+data_info['n_users']+2])

    user_fea_index = torch.tensor(user_fea_index).to(device)

    return user_fea_index

# ====================Sampling=================================
def pickle_array(array, path):
    with open(path, "wb") as fout:
        pickle.dump(array, fout, pickle.HIGHEST_PROTOCOL)

def load_pickled_array(path):
    with open(path, "rb") as fin:
        return pickle.load(fin)

def sampling_block(users, item_num, neg_ratio, interacted_items, replace=False, sampling_sift_pos=False, seed=None, dump_path=None):
    if seed is not None:
        np.random.seed(seed)

    neg_candidates = np.arange(item_num)
    if sampling_sift_pos:
        neg_items = []
        for u in users:
            probs = np.ones(item_num)
            probs[interacted_items[str(int(u))]] = 0
            probs /= np.sum(probs)

            u_neg_items = np.random.choice(neg_candidates, size=neg_ratio, p=probs, replace=replace).reshape(1, -1)
            neg_items.append(u_neg_items)
        neg_items = np.concatenate(neg_items, axis=0)
    else:
        neg_items = np.random.choice(neg_candidates, (len(users), neg_ratio), replace=replace)
    if dump_path is not None:
        pickle_array(neg_items, dump_path)

    neg_items = torch.from_numpy(neg_items)

    return neg_items

def negative_sampling(args, users, model, user_his_iid):
    if args.neg_num > 0:
        if args.sampling_num_process > 1:
            chunked_query_indexes = np.array_split(users, args.sampling_num_process)
            if args.fix_sampling_seeds:
                seeds = np.random.randint(1000000, size=args.sampling_num_process)
            else:
                seeds = [None] * args.sampling_num_process
            pool = mp.Pool(args.sampling_num_process)
            block_result = []
            os.makedirs("./tmp/pid_{}/".format(os.getpid()), exist_ok=True)
            dump_paths = ["./tmp/pid_{}/part_{}.pkl".format(os.getpid(), idx) for idx in
                          range(len(chunked_query_indexes))]
            for idx, block_query_indexes in enumerate(chunked_query_indexes):
                pool.apply_async(sampling_block, args=(block_query_indexes,
                                                       model.num_items,
                                                       args.neg_num,
                                                       user_his_iid,
                                                       args.replace,
                                                       args.sampling_sift_pos,
                                                       seeds[idx],
                                                       dump_paths[idx]))
            pool.close()
            pool.join()
            block_result = [load_pickled_array(dump_paths[idx]) for idx in range(len(chunked_query_indexes))]
            shutil.rmtree("./tmp/pid_{}/".format(os.getpid()))
            neg_items = np.vstack(block_result)
        else:
            neg_items = sampling_block(users,
                                       model.num_items,
                                       args.neg_num,
                                       user_his_iid,
                                       args.replace,
                                       args.sampling_sift_pos)
        return neg_items

# ====================Metrics==============================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k: top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall_n = np.where(recall_n != 0, recall_n, 1)
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n

    return {'recall': recall, 'precision': precis}

def MRRatK_r(test_data, r, k):
    """
    Mean Reciprocal Rank
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]
    scores = np.arange(1, k + 1)
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)
