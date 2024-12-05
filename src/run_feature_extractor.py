import time
import argparse
import tqdm as tqdm
from utils import *
import torch.optim as optim
from dataloader import get_dataloader
from collections import defaultdict, OrderedDict
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel, AutoConfig

class Wav2Vec2ClassificationModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config, hidden_size, rec_dim, data_info, item_fea_index, item_fea_mask, dropout):
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)
        self.hidden_size = hidden_size
        self.fc = nn.Linear(config.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gender_fc = nn.Linear(self.hidden_size, 2)
        self.age_fc = nn.Linear(self.hidden_size, 3)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.rec_dim = rec_dim
        self.datainfo = data_info
        self.item_fea_index = item_fea_index
        self.item_fea_mask = item_fea_mask
        self.spk_fc = nn.Linear(self.hidden_size, self.rec_dim)
        self.num_users = self.datainfo['n_users']
        self.num_items = self.datainfo['n_items']
        self.n_item_fea = self.datainfo['n_item_fea']
        self.n_user_fea = 5  # gender2 + age3
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.n_user_fea,
                                                 embedding_dim=self.rec_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items + self.n_item_fea,
                                                 embedding_dim=self.rec_dim)

        self.init_weights()

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(self, hidden_states):
        outputs = torch.mean(hidden_states, dim=1)
        return outputs

    def get_item_emb(self, items):
        if args.dataset == 'coat':
            item_fea_emb = self.embedding_item(self.item_fea_index[items])
        elif args.dataset == 'movielens1m':
            item_fea_emb = self.embedding_item(self.item_fea_index[items] * self.item_fea_mask[items])
        return torch.mean(item_fea_emb, dim=-2)

    def getRating(self, sph_emb):
        items = torch.arange(self.num_items).to(args.device)
        item_embeds = self.get_item_emb(items)
        return torch.matmul(sph_emb, item_embeds.to(sph_emb).t())

    def forward(
            self,
            input_values,
            users,
            pos_items,
            neg_items=0,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        with torch.no_grad():
            outputs = self.wav2vec2(
                input_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        hidden_states = outputs[0]
        x = self.merged_strategy(hidden_states)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)

        gender_logits = self.gender_fc(x)
        gender_logits = F.log_softmax(gender_logits, dim=-1)
        age_logits = self.age_fc(x)
        age_logits = F.log_softmax(age_logits, dim=-1)

        sph_emb = self.spk_fc(x)

        pos_item_emb = self.get_item_emb(pos_items)
        neg_item_emb = self.get_item_emb(neg_items)

        return age_logits, gender_logits, sph_emb, pos_item_emb, neg_item_emb

def norm_loss(model):
    loss = 0.0
    for name, param in model.named_parameters():
        if 'embedding' in name:
            loss += torch.sum(param ** 2)
    return loss / 2

def train_single_epoch(model, dataloader, optimizer, device, user_his_iid):
    model.train()
    losses = []
    for user, pos_item, waveform, age_label, gender_label in tqdm.tqdm(dataloader):
        neg_item = negative_sampling(args, user, model, user_his_iid)

        user = user.to(device)
        pos_item = pos_item.to(device)
        neg_item = neg_item.to(device)
        waveform = waveform.to(device)
        age_label = age_label.to(device)
        gender_label = gender_label.to(device)

        age_logits, gender_logits, sph_emb, pos_item_emb, neg_item_emb = model(waveform, user, pos_item, neg_item)

        age_loss = F.nll_loss(age_logits, age_label)
        gender_loss = F.nll_loss(gender_logits, gender_label)

        pos_pred = cosine_sim(sph_emb, pos_item_emb)
        neg_pred = cosine_sim(sph_emb, neg_item_emb)
        rec_pred = torch.cat((pos_pred, neg_pred), dim=1)

        rec_loss = args.rec_weight * loss_fcn(rec_pred)
        reg_loss = norm_loss(model)
        loss = age_loss + gender_loss + rec_loss + args.weight_decay * reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    torch.save(clf_model, f'./spk_checkpoints/clf_model_{args.dataset}.pt')
    return losses

def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.eq(target).sum().item()

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

def test_val_single_batch(X, k):
    sorted_items = X[0]
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    ret = RecallPrecision_ATk(groundTrue, r, k)
    return ret['precision'], ret['recall'], NDCGatK_r(groundTrue,r,k)

def test_val_single_epoch(model, dataloader, device, topk, mask, user_his_iid, multicore=0):
    model.eval()

    age_correct = 0
    gender_correct = 0

    rating_list = []
    groundTrue_list = []
    sph_emb_collect = defaultdict(list)

    mask = mask.to(device)

    with torch.no_grad():
        for test_user, test_item, waveform, age_label, gender_label in tqdm.tqdm(dataloader):
            test_user = test_user.to(device)
            test_item = test_item.to(device)
            waveform = waveform.to(device)
            age_label = age_label.to(device)
            gender_label = gender_label.to(device)

            age_logits, gender_logits, sph_embs, _, _ = model(waveform, test_user, test_item)

            age_pred = get_likely_index(age_logits)
            gender_pred = get_likely_index(gender_logits)

            age_correct += number_of_correct(age_pred, age_label)
            gender_correct += number_of_correct(gender_pred, gender_label)

            for key, emb in zip(test_user, sph_embs):
                sph_emb_collect[int(key)].append(emb)

        for key, value in sph_emb_collect.items():
            sph_emb_collect[key] = torch.mean(torch.stack(value, dim=0), dim=0)
            rating = model.getRating(sph_emb_collect[key])
            # rating = rating.cpu()
            rating += mask[key]

            _, rating_K = torch.topk(rating, k=topk)
            rating_list.append(rating_K)

            groundTrue_list.append(user_his_iid[str(int(key))])

    age_accu = age_correct / len(dataloader.dataset) * 100.
    gender_accu = gender_correct / len(dataloader.dataset) * 100.

    rating_list = [rating_list[i:i+args.batch_size] for i in range(0, model.num_users, args.batch_size)]
    groundTrue_list = [groundTrue_list[i:i+args.batch_size] for i in range(0, model.num_users, args.batch_size)]

    X = zip(rating_list, groundTrue_list)
    Recall, Precision, NDCG = 0, 0, 0

    for i, x in enumerate(X):
        precision, recall, ndcg = test_val_single_batch(x, topk)
        Recall += recall
        Precision += precision
        NDCG += ndcg

    Precision /= model.num_users
    Recall /= model.num_users
    NDCG /= model.num_users
    if Precision + Recall != 0:
        F1_score = 2 * (Precision * Recall) / (Precision + Recall)
    else:
        F1_score = 0

    return age_accu, gender_accu, F1_score, Precision, Recall, NDCG

def train(model, train_loader, val_loader, optimizer, device, mask, user_his_train, user_his_val):
    for epoch in tqdm.tqdm(range(args.epochs)):
        losses = train_single_epoch(model, train_loader, optimizer, device, user_his_train)
        loss = np.mean(losses)

        age_accu, gender_accu, F1_score, Precision, Recall, NDCG = test_val_single_epoch(model, val_loader, device, args.topk, mask, user_his_val, args.test_multicore)

        print(f"\nTrain Epoch: {epoch + 1} Loss: {loss:.6f} Age: {age_accu:.2f}% Gender: {gender_accu:.2f}%")
        print(f"\nF1_score: {F1_score:6f} Precision: {Precision:.6f} Recall: {Recall:.6f} NDCG: {NDCG:.6f}")
    print('Finished Training')

def user_labels_emb_output(train_loader, device):
    model = torch.load(f'./spk_checkpoints/clf_model_{args.dataset}.pt')
    model = model.to(device)

    model.eval()
    user_emb_dict = defaultdict(list)
    item_emb_dict = defaultdict(list)
    emb_sum = defaultdict(list)

    user_pred_labels = defaultdict(list)
    with torch.no_grad():
        for users, items, waveforms, _, _ in tqdm.tqdm(train_loader):
            users = users.to(device)
            items = items.to(device)
            waveforms = waveforms.to(device)

            age_logits, gender_logits, sph_embs, item_embs, _ = model(waveforms, users, items)

            # append pred user label
            age_preds = get_likely_index(age_logits)
            gender_preds = get_likely_index(gender_logits)

            for user, gender_pred, age_pred in zip(users, gender_preds, age_preds):
                user_pred_labels[int(user)].append([int(gender_pred), int(age_pred)])

            # append emb
            for user, item, sph_emb, item_emb in zip(users, items, sph_embs, item_embs):
                user_emb_dict[int(user)].append(sph_emb)
                item_emb_dict[int(item)].append(item_emb)

                emb_sum[int(user)].append(item_emb)

    # save user info labels
    user_pred_labels = OrderedDict(sorted(user_pred_labels.items(), key=lambda x: int(x[0])))

    with open(f'./user_pred_labels_{args.dataset}.json', 'w') as f:
        json.dump(user_pred_labels, f, indent=4)

    print(f'User labels stored...')

    # save emb
    for key, value in user_emb_dict.items():
        user_emb_dict[key] = torch.mean(torch.stack(value, dim=0), dim=0)

        emb_sum[key] = torch.mean(torch.stack(emb_sum[key], dim=0), dim=0)
        emb_sum[key] = (torch.mul(user_emb_dict[key], args.ens_ratio) + torch.mul(emb_sum[key], (1-args.ens_ratio))).tolist()

    for key, value in item_emb_dict.items():
        item_emb_dict[key] = torch.mean(torch.stack(value, dim=0), dim=0).tolist()

    emb_sum = OrderedDict(sorted(emb_sum.items(), key=lambda x: int(x[0])))

    with open(f'./emb/emb_sum_{args.dataset}.json', 'w') as f:
        json.dump(emb_sum, f, indent=4)

    print(f'Embedding stored...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='speech classification')
    parser.add_argument('--batch_size',
                        type=int,
                        default=4)
    parser.add_argument('--hidden_size',
                        type=int,
                        default=1024)
    parser.add_argument('--dropout',
                        type=float,
                        default=0.2)
    parser.add_argument('--recdim',
                        type=int,
                        default=64)
    parser.add_argument('--seed',
                        type=int,
                        default=2023)
    parser.add_argument('--epochs',
                        type=int,
                        default=20)
    parser.add_argument('--lr',
                        type=float,
                        default=0.0005)
    parser.add_argument('--train',
                        type=int,
                        default=1)
    parser.add_argument('--test',
                        type=int,
                        default=1)
    parser.add_argument('--dataset',
                        type=str,
                        default='coat')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0')

    parser.add_argument('--weight_decay',
                        type=float,
                        default=1e-4)
    parser.add_argument('--rec_weight',
                        type=float,
                        default=1.0)

    parser.add_argument('--topk',
                        type=int,
                        default=10)
    parser.add_argument('--test_multicore',
                        type=int,
                        default=0,
                        help='whether we use multiprocessing or not in test')
    parser.add_argument('--ens_ratio',
                        type=float,
                        default=0.5)

    parser.add_argument('--neg_num',
                        type=int,
                        default=1)
    parser.add_argument('--sampling_num_process',
                        type=int,
                        default=1)
    parser.add_argument('--fix_sampling_seeds',
                        action='store_false')
    parser.add_argument('--sampling_sift_pos',
                        action='store_true',
                        help='whether to sift the pos item when doing negative sampling')
    parser.add_argument('--replace',
                        action='store_false',
                        help='')

    args = parser.parse_args()

    init_seed(args.seed)

    if torch.cuda.is_available():
        device = args.device
        torch.backends.cudnn.benchmark = False
    else:
        device = 'cpu'

    with open(f'../data/{args.dataset}/user_his_train.json', 'r') as f:
        user_his_train = json.load(f)
    with open(f'../data/{args.dataset}/user_his_val.json', 'r') as f:
        user_his_val = json.load(f)
    with open(f'../data/{args.dataset}/user_his_test.json', 'r') as f:
        user_his_test = json.load(f)

    if args.dataset == 'coat':
        data_info, item_fea_index = get_data_info(args.dataset, device)
        item_fea_mask = None
    elif args.dataset == 'movielens1m':
        data_info, item_fea_index, item_fea_mask = get_data_info(args.dataset, device)

    mask = torch.zeros(data_info['n_users'], data_info['n_items'])
    for key, value in user_his_train.items():
        mask[int(key)][value] = -np.inf

    train_loader, val_loader, test_loader = get_dataloader(args, 'spk', data_info, device)

    model_name_or_path = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        finetuning_task="wav2vec2_clf",
    )

    clf_model = Wav2Vec2ClassificationModel.from_pretrained(
        model_name_or_path,
        config=config,
        hidden_size=args.hidden_size,
        rec_dim=args.recdim,
        data_info=data_info,
        item_fea_index=item_fea_index,
        item_fea_mask=item_fea_mask,
        dropout=args.dropout,
    )

    clf_model = clf_model.to(device)
    clf_model.freeze_feature_extractor()
    loss_fcn = CosineContrastiveLoss(margin=0.3)
    if args.dataset == 'movielens1m':
        loss_fcn = CosineContrastiveLoss(margin=0.1)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, clf_model.parameters()), lr=args.lr)

    os.makedirs(f'./spk_checkpoints', exist_ok=True)
    if args.train:
        train(clf_model, train_loader, val_loader, optimizer, device, mask, user_his_train, user_his_val)

    if args.test:
        test_model = torch.load(f'./spk_checkpoints/clf_model_{args.dataset}.pt')
        age_accu, gender_accu, F1_score, Precision, Recall, NDCG = test_val_single_epoch(test_model, test_loader, device, args.topk, mask, user_his_test, args.test_multicore)
        torch.save(clf_model, f'./spk_checkpoints/clf_model_{args.dataset}.pt')
        print(f"Test set: Age: {age_accu:.2f}% Gender: {gender_accu:.2f}%")
        print(f"\nF1_score: {F1_score:6f} Precision: {Precision:.6f} Recall: {Recall:.6f} NDCG: {NDCG:.6f}")

    user_labels_emb_output(train_loader, device)