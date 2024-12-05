import argparse
import time
import tqdm
import csv
import torch.optim as optim
from dataloader import get_dataloader
from model import GIP
from utils import *

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
    return ret['precision'], ret['recall'], NDCGatK_r(groundTrue,r,k)# , MRRatK_r(groundTrue, r, k)

def test_val_single_epoch(model, dataloader, device, topk, mask, user_his_iid, multicore=0):
    model.eval()

    users_list = []
    rating_list = []
    groundTrue_list = []

    mask = mask.to(device)

    with torch.no_grad():
        for test_user in tqdm.tqdm(dataloader):
            test_user = test_user.to(device)

            rating = model.getRating(test_user)
            # rating = rating.cpu()
            rating += mask[test_user]

            _, rating_K = torch.topk(rating, k=topk)
            rating_list.append(rating_K)

            groundTrue_list.append([user_his_iid[str(int(u))] for u in test_user])

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

    return Precision, Recall, NDCG
# F1_score, 
def train_single_epoch(model, dataloader, optimizer, device, user_his_iid):
    model.train()
    losses = []
    for user, pos_item in tqdm.tqdm(dataloader):
        neg_item = negative_sampling(args, user, model, user_his_iid)

        user = user.to(device)
        pos_item = pos_item.to(device)
        neg_item = neg_item.to(device)

        user_emb, pos_item_emb, neg_item_emb, user_emb_ego, pos_item_ego, neg_item_ego = model.get_all_emb(user, pos_item, neg_item)

        reg_loss = (1/2) * (user_emb_ego.norm(2).pow(2) + pos_item_ego.norm(2).pow(2) + neg_item_ego.norm(2).pow(2)) / float(len(user))

        pos_pred = cosine_sim(user_emb, pos_item_emb)
        neg_pred = cosine_sim(user_emb, neg_item_emb)
        rec_pred = torch.cat((pos_pred, neg_pred), dim=1)
        rec_loss = loss_fcn(rec_pred)

        loss = rec_loss + args.weight_decay * reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

def train(model, train_loader, val_loader, optimizer, stopper, device, mask, user_his_train, user_his_val):
    for epoch in tqdm.tqdm(range(args.epochs)):
        losses = train_single_epoch(model, train_loader, optimizer, device, user_his_train)
        loss = np.mean(losses)
        # F1_score, F1_score: {F1_score:6f} 
        Precision, Recall, NDCG = test_val_single_epoch(model, val_loader, device, args.topk, mask, user_his_val, args.test_multicore)

        print(f"\nTrain Epoch: {epoch + 1} Training Loss: {loss:.6f}")
        print(f"\nPrecision: {Precision:.6f} Recall: {Recall:.6f} NDCG: {NDCG:.6f}")

        # early stopping
        stopper(Precision, model=model, args=args)
        if stopper.early_stop:
            print('Early stopping!')
            print(f'Best precision is:{stopper.best_score}')
            break
    print('Finished Training')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2023, help="the random seed")
    parser.add_argument('--dataset', type=str, default='coat', help="dataset")
    parser.add_argument('--epochs', type=int, default=1000, help="")
    parser.add_argument('--batch_size', type=int, default=512, help="")
    parser.add_argument('--lr', type=float, default=0.001, help="")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="the weight decay of optimizer")
    parser.add_argument('--topk', type=int, default=10, help="")

    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--recdim', type=int, default=64, help="the latent vector embedding size")
    parser.add_argument('--n_layers', type=int, default=3, help="the graph convolution layer num")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--g_drop', action='store_true')
    parser.add_argument('--keepprob', type=float, default=0.6)
    parser.add_argument('--emb_dropout', type=float, default=0.0)

    parser.add_argument('--test', type=int, default=1)
    parser.add_argument('--test_multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--ens_ratio', type=float, default=0.5, help='the aggregation ratio of speaker embedding')

    parser.add_argument('--neg_num', type=int, default=1)
    parser.add_argument('--sampling_num_process', type=int, default=1)
    parser.add_argument('--fix_sampling_seeds', action='store_false')
    parser.add_argument('--sampling_sift_pos', action='store_true', help='whether to sift the pos item when doing negative sampling')
    parser.add_argument('--replace', action='store_false', help='')

    args = parser.parse_args()

    # init random seed
    init_seed(args.seed)

    # define device
    if torch.cuda.is_available():
        device = args.device
        torch.backends.cudnn.benchmark = False
    else:
        device = 'cpu'

    # load user history
    with open(f'../data/{args.dataset}/user_his_train.json', 'r') as f:
        user_his_train = json.load(f)
    with open(f'../data/{args.dataset}/user_his_val.json', 'r') as f:
        user_his_val = json.load(f)
    with open(f'../data/{args.dataset}/user_his_test.json', 'r') as f:
        user_his_test = json.load(f)

    # load data info
    if args.dataset == 'coat':
        data_info, item_fea_index = get_data_info(args.dataset, device)
        item_fea_mask = None
    elif args.dataset == 'movielensmini':
        data_info, item_fea_index, item_fea_mask = get_data_info(args.dataset, device)
    elif args.dataset == 'movielens1m':
        data_info, item_fea_index, item_fea_mask = get_data_info(args.dataset, device)

    user_fea_index = get_user_fea_index(args.dataset, data_info, device)

    mask = torch.zeros(data_info['n_users'], data_info['n_items'])
    for key, value in user_his_train.items():
        mask[int(key)][value] = -np.inf

    # dataloader
    train_loader, val_loader, test_loader, sp_graph, spk_emb_list = get_dataloader(args, 'gnn', data_info, device)

    # model
    model = GIP(args, data_info, user_fea_index, item_fea_index, item_fea_mask, sp_graph, spk_emb_list).to(device)
    loss_fcn = CosineContrastiveLoss(margin=0.3)
    if args.dataset == 'movielensmini':
        loss_fcn = CosineContrastiveLoss(margin=0.1)
    if args.dataset == 'movielens1m':
        loss_fcn = CosineContrastiveLoss(margin=0.1)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    stopper = EarlyStopping(patience=args.patience, verbose=True)

    os.makedirs(f'./checkpoints', exist_ok=True)
    train(model, train_loader, val_loader, optimizer, stopper, device, mask, user_his_train, user_his_val)

    if args.test:
        best_model = torch.load(f'./checkpoints/{args.dataset}.pt', weights_only=False)
        Precision, Recall, NDCG = test_val_single_epoch(best_model, test_loader, device, args.topk, mask, user_his_test, args.test_multicore)
        print(f'Test results:')
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        print(f"F1: {F1:.6f} Precision: {Precision:.6f} Recall: {Recall:.6f} NDCG: {NDCG:.6f}")
