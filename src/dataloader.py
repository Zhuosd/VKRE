import os
import json
import torch
import logging
import torchaudio
import numpy as np
from time import time
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

#  AudioDataset------------------------------
class AudioDataset(Dataset):
    def __init__(self, datalist, audio_path, target_sample_rate, transformation=None):
        self.datalist = datalist
        self.audio_path = audio_path
        self.target_sample_rate = target_sample_rate
        self.transformation = None
        if transformation:
            self.transformation = transformation

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        user_id = self.datalist[idx][0]
        item_id = self.datalist[idx][1]
        audio_file_path = os.path.join(self.audio_path, self.datalist[idx][2])
        audio_name = self.datalist[idx][2]
        age_label, gender_label = self._get_label(audio_name)
        waveform, sample_rate = torchaudio.load(audio_file_path)
        if sample_rate != self.target_sample_rate:
            waveform = self._resample(waveform, sample_rate)

        return user_id, item_id, waveform, age_label, gender_label

    def _get_label(self, audio_name):
        name_list = audio_name.split('_')
        age_labels = {'under 20': 0, '20-30': 1, 'over 30': 2}
        gender_labels = {'women': 0, 'men': 1}

        age = age_labels[name_list[-3]]
        gender = gender_labels[name_list[-2]]

        return age, gender

    def _resample(self, waveform, sample_rate):
        resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)

        return resampler(waveform)

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros, aim at audios
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def clf_collate_fn(batch):
    users, items, audios, age_labels, gender_labels = [], [], [], [], []

    for user, item, waveform, age, gender in batch:
        users += [torch.tensor(int(user))]
        items += [torch.tensor(int(item))]
        audios += [waveform]
        age_labels += [torch.tensor(age)]
        gender_labels += [torch.tensor(gender)]

    users = torch.stack(users)
    items = torch.stack(items)
    audios = pad_sequence(audios)
    age_labels = torch.stack(age_labels)
    gender_labels = torch.stack(gender_labels)

    return users, items, audios.squeeze(dim=1), age_labels, gender_labels

#  GraphDataset------------------------------
class GraphDataset(Dataset):
    def __init__(self, datalist):
        self.datalist = datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        user_id = self.datalist[idx][0]
        item_id = self.datalist[idx][1]

        return user_id, item_id

def gcn_collate_fn(batch):
    users, items = [], []

    for user, item in batch:
        users += [torch.tensor(int(user))]
        items += [torch.tensor(int(item))]

    users = torch.stack(users)
    items = torch.stack(items)

    return users, items

def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

def getSparseGraph(train_set, path, data_info, device):
    print("generating adjacency matrix")
    num_users, num_items = data_info['n_users'], data_info['n_items']
    trainUser, trainItem = train_set[:, 0], train_set[:, 1]
    UserItemNet = csr_matrix((np.ones(len(trainUser)), (trainUser, trainItem)),
                                  shape=(num_users, num_items))
    users_D = np.array(UserItemNet.sum(axis=1)).squeeze()
    users_D[users_D == 0.] = 1.
    items_D = np.array(UserItemNet.sum(axis=0)).squeeze()
    items_D[items_D == 0.] = 1.

    adj_mat = sp.dok_matrix((num_users + num_items, num_users + num_items),
                            dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = UserItemNet.tolil()
    adj_mat[:num_users, num_users:] = R
    adj_mat[num_users:, :num_users] = R.T
    adj_mat = adj_mat.todok()
    # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

    rowsum = np.array(adj_mat.sum(axis=1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)

    norm_adj = d_mat.dot(adj_mat)
    norm_adj = norm_adj.dot(d_mat)
    norm_adj = norm_adj.tocsr()

    Graph = _convert_sp_mat_to_sp_tensor(norm_adj)
    Graph = Graph.coalesce().to(device)
    return Graph

# get_dataloader-----------------------------
def get_dataloader(args, task, data_info, device):
    train_set, val_set, test_set = [], [], []

    with open(f"../data/{args.dataset}/train.json", 'r') as train_f:
        train_audio = json.load(train_f)
    for key, value in train_audio.items():
        for iid, mp3 in value:
            train_set.append([key, iid, mp3])

    with open(f"../data/{args.dataset}/val.json", 'r') as val_f:
        val_audio = json.load(val_f)
    for key, value in val_audio.items():
        for iid, mp3 in value:
            val_set.append([key, iid, mp3])

    with open(f"../data/{args.dataset}/test.json", 'r') as test_f:
        test_audio = json.load(test_f)
    for key, value in test_audio.items():
        for iid, mp3 in value:
            test_set.append([key, iid, mp3])

    if args.device[:4] == "cuda":
        num_workers = 0
        pin_memory = False
    else:
        num_workers = 0
        pin_memory = False

    if task == 'spk':
        input_dir = f'../data/{args.dataset}/mp3/'

        train_set = AudioDataset(train_set, input_dir, 16000)
        val_set = AudioDataset(val_set, input_dir, 16000)
        test_set = AudioDataset(test_set, input_dir, 16000)

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=clf_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=clf_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=clf_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return train_loader, val_loader, test_loader

    elif task == 'gnn':
        train_set = np.array(train_set)[:, :-1]
        val_set = np.array(val_set)[:, :-1]
        test_set = np.array(test_set)[:, :-1]

        sp_graph = getSparseGraph(train_set, f'../data/{args.dataset}/', data_info, device)

        train_set = GraphDataset(train_set)
        val_set = GraphDataset(val_set)
        test_set = GraphDataset(test_set)

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=gcn_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        val_loader = DataLoader(
            list(range(data_info['n_users'])),
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        test_loader = DataLoader(
            list(range(data_info['n_users'])),
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        with open(f'./emb/emb_sum_{args.dataset}.json', 'r') as f:
            spk_meb_dict = json.load(f)
        spk_emb_list = torch.as_tensor(list(spk_meb_dict.values())).to(device)

        return train_loader, val_loader, test_loader, sp_graph, spk_emb_list



