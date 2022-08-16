import argparse
import os
from utils import eval_by_ts
import pandas as pd
import uuid

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
parser.add_argument('--model_name', type=str, default='', help='name of stored model')
parser.add_argument('--base_model', type=str, choices=['origin', 'np', 'snp', 'anp', 'mnp'], default='origin',
                    help='base model structure')
parser.add_argument('--ode', action='store_true', help='enable ODE decoder')
parser.add_argument('-d', '--determinstic', action='store_true', help='enable determinstic path decoder')
parser.add_argument('--resize_ratio', default=0, type=int, help='timestamp resize ratio')
parser.add_argument('--rand_edge_features', type=int, default=0, help='use random edge featrues')
parser.add_argument('--rand_node_features', type=int, default=0, help='use random node featrues')
parser.add_argument('--eval_neg_samples', type=int, default=1, help='how many negative samples to use at inference. Note: this will change the metric of test set to AP+AUC to AP+MRR!')
parser.add_argument('--zero_feat', action='store_true', help='use all 0 featrues')
parser.add_argument('--train_ratio', default=0, type=float, help='precentage of training ratio')
parser.add_argument('--seed', default=123, type=int, help='random seed')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import time
import random
import dgl
import numpy as np
from modules import *
from sampler import *
from utils import *
from sklearn.metrics import average_precision_score, roc_auc_score
from neural_process import NeuralProcess
from torch.distributions.kl import kl_divergence
from tqdm import tqdm
import datetime


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(args.seed)

g, df = load_graph(args.data)
node_feats, edge_feats = load_feat(args.data, g['indptr'].shape[0] - 1, len(df), args.rand_edge_features,
                                   args.rand_node_features, args.zero_feat)
sample_param, memory_param, gnn_param, np_param, train_param = parse_config(args.config)
train_edge_end = df[df['ext_roll'].gt(0)].index[0]
val_edge_end = df[df['ext_roll'].gt(1)].index[0]

gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]
combine_first = False
if 'combine_neighs' in train_param and train_param['combine_neighs']:
    combine_first = True
# Register Neural Process
if args.resize_ratio == 0:
    resize_ratio = 10 ** (len(str(int(df['time'].max()))) - 1)
    print(f"Resize ratio is automatically set to: {resize_ratio}")
else:
    resize_ratio = args.resize_ratio
model = NeuralProcess(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param, np_param,
                      args.base_model, args.ode, args.determinstic, resize_ratio,
                      combined=combine_first).cuda()
mailbox = MailBox(memory_param, g['indptr'].shape[0] - 1, gnn_dim_edge) if memory_param['type'] != 'none' else None
creterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])
if 'all_on_gpu' in train_param and train_param['all_on_gpu']:
    if node_feats is not None:
        node_feats = node_feats.cuda()
    if edge_feats is not None:
        edge_feats = edge_feats.cuda()
    if mailbox is not None:
        mailbox.move_to_gpu()

sampler = None
if not ('no_sample' in sample_param and sample_param['no_sample']):
    sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                              sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                              sample_param['strategy'] == 'recent', sample_param['prop_time'],
                              sample_param['history'], float(sample_param['duration']))
neg_link_sampler = NegLinkSampler(g['indptr'].shape[0] - 1)


def eval(mode='val', stage='val'):
    neg_samples = 1
    model.eval()
    y_pred_list = list()
    y_label_list = list()
    ts_list = list()
    ap_list = list()
    auc_list = list()
    aucs_mrrs = list()
    mrr_ts = list()
    if mode == 'val':
        eval_df = df[train_edge_end:val_edge_end]
        if stage == 'test' and args.base_model != 'origin':
            neg_samples = train_param['train_neg_size']
            batch_size = train_param['history_encoding_batch_size']
        else:
            neg_samples = args.eval_neg_samples
            batch_size = train_param['test_batch_size']
    elif mode == 'test':
        eval_df = df[val_edge_end:]
        neg_samples = args.eval_neg_samples
        batch_size = train_param['test_batch_size']
    elif mode == 'train':
        eval_df = train_data
        if stage == 'test' and args.base_model != 'origin':
            batch_size = train_param['history_encoding_batch_size']
            neg_samples = train_param['train_neg_size']
        else:
            batch_size = train_param['batch_size']
    with torch.no_grad():
        total_loss = 0
        for _, rows in eval_df.groupby(eval_df.index // batch_size):
            root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows) * neg_samples)]).astype(np.int32)
            ts = np.tile(rows.time.values, neg_samples + 2).astype(np.float32)
            if sampler is not None:
                if 'no_neg' in sample_param and sample_param['no_neg']:
                    pos_root_end = len(rows) * 2
                    sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                else:
                    sampler.sample(root_nodes, ts)
                ret = sampler.get_ret()
            if gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, sample_param['history'])
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts)
            mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0])
            pred_pos, pred_neg = model(mfgs, ts, neg_samples=neg_samples, data=mode)
            total_loss += creterion(pred_pos, torch.ones_like(pred_pos))
            total_loss += creterion(pred_neg, torch.zeros_like(pred_neg))
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).view(-1).sigmoid().cpu().numpy()
            y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0).view(-1).numpy()
            y_pred_list.append(y_pred)
            y_label_list.append(y_true)
            ap_list.append(average_precision_score(y_true, y_pred))
            auc_list.append(roc_auc_score(y_true, y_pred))
            ts_list.append(np.tile(rows.time.values, neg_samples + 1).astype(np.float32))
            if neg_samples > 1:
                aucs_mrrs.append(torch.reciprocal(torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(neg_samples, -1), dim=0) + 1).type(torch.float))
                mrr_ts.append(rows.time.values)
            if mailbox is not None:
                eid = rows['Unnamed: 0'].values
                mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                block = None
                if memory_param['deliver_to'] == 'neighbors':
                    block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
                mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block, neg_samples=neg_samples)
                mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts, neg_samples=neg_samples)
        if mode == 'val':
            val_losses.append(float(total_loss))
    y_pred = np.concatenate(y_pred_list)
    y_true = np.concatenate(y_label_list)
    all_ts = np.concatenate(ts_list)
    ap = average_precision_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    auc_mrr = -1
    if neg_samples > 1:
            auc_mrr = float(torch.cat(aucs_mrrs).mean())
    if mode == "test":
        raw_results = pd.DataFrame({'y_pred': y_pred, 'y_true': y_true, 't': all_ts})
        raw_mrr_results = None
        if neg_samples > 1:
            raw_mrr = torch.cat(aucs_mrrs).cpu().numpy()
            mrr_ts = np.concatenate(mrr_ts)
            raw_mrr_results =pd.DataFrame({'mrr': raw_mrr, 't': mrr_ts})
        old_ap = float(torch.tensor(ap_list).mean())
        old_auc = float(torch.tensor(auc_list).mean())
        print(f"Old AP: {old_ap}, Old AUC: {old_auc}")
        return ap, auc, auc_mrr, [raw_results, raw_mrr_results]
    else:
        return ap, auc, auc_mrr


if not os.path.isdir('models'):
    os.mkdir('models')
dt = datetime.datetime.now()
if args.model_name == '':
    rand_suffix = str(uuid.uuid4()).split('-')[0]
    path_saver = 'models/{}_{}_{:02d}-{:02d}-{:02d}_{}.pkl'.format(args.data, dt.date(), dt.hour, dt.minute, dt.second, rand_suffix)
else:
    path_saver = 'models/{}_{}_{:02d}-{:02d}-{:02d}.pkl'.format(args.model_name, dt.date(), dt.hour, dt.minute,
                                                                dt.second)
best_ap = 0
best_e = 0
early_stop_counter = 0
train_neg_samples = train_param.get('train_neg_size', 1)
val_losses = list()
group_indexes = list()
if args.train_ratio > 0 and args.train_ratio < 1:
    train_data = df[:train_edge_end].sample(frac=args.train_ratio, random_state=args.seed).sort_index(ignore_index=True)
elif args.train_ratio > 1:
    train_data = df[:train_edge_end].sample(n=args.train_ratio, random_state=args.seed).sort_index(ignore_index=True)
else:
    train_data = df[:train_edge_end]
group_indexes.append(np.array(train_data.index // train_param['batch_size']))
if 'reorder' in train_param:
    # random chunk shceduling
    reorder = train_param['reorder']
    group_idx = list()
    for i in range(reorder):
        group_idx += list(range(0 - i, reorder - i))
    group_idx = np.repeat(np.array(group_idx), train_param['batch_size'] // reorder)
    group_idx = np.tile(group_idx, train_edge_end // train_param['batch_size'] + 1)[:train_edge_end]
    group_indexes.append(group_indexes[0] + group_idx)
    base_idx = group_indexes[0]
    for i in range(1, train_param['reorder']):
        additional_idx = np.zeros(train_param['batch_size'] // train_param['reorder'] * i) - 1
        group_indexes.append(np.concatenate([additional_idx, base_idx])[:base_idx.shape[0]])
for e in range(train_param['epoch']):
    print('Epoch {:d}:'.format(e))
    time_sample = 0
    time_prep = 0
    time_tot = 0
    total_loss = 0
    total_nll_loss = 0
    total_kl_loss = 0
    batch_num = 0
    # training
    model.train()
    model.reset()
    if sampler is not None:
        sampler.reset()
    if mailbox is not None:
        mailbox.reset()
        model.memory_updater.last_updated_nid = None
    total_batch_num = len(train_data.groupby(group_indexes[random.randint(0, len(group_indexes) - 1)]))
    for i, rows in tqdm(train_data.groupby(group_indexes[random.randint(0, len(group_indexes) - 1)])):
        t_tot_s = time.time()
        batch_num += 1
        root_nodes = np.concatenate(
            [rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows) * train_neg_samples)]).astype(np.int32)
        ts = np.tile(rows.time.values, train_neg_samples + 2).astype(np.float32)
        if sampler is not None:
            if 'no_neg' in sample_param and sample_param['no_neg']:
                pos_root_end = root_nodes.shape[0] * 2 // 3
                sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
            else:
                sampler.sample(root_nodes, ts)
            ret = sampler.get_ret()
            time_sample += ret[0].sample_time()
        t_prep_s = time.time()
        if gnn_param['arch'] != 'identity':
            mfgs = to_dgl_blocks(ret, sample_param['history'])
        else:
            mfgs = node_to_dgl_blocks(root_nodes, ts)
        mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
        if mailbox is not None:
            mailbox.prep_input_mails(mfgs[0])
        time_prep += time.time() - t_prep_s
        optimizer.zero_grad()
        if args.determinstic:
            pred_pos, pred_neg, dist = model(mfgs, ts, train_neg_samples)
            y_target = torch.cat([torch.ones_like(pred_pos), torch.zeros_like(pred_neg)])
            loss = -dist.log_prob(y_target).sum(-1).mean()
            nll = loss
        else:
            pred_pos, pred_neg, q_target, q_context = model(mfgs, ts, train_neg_samples)
            nll = creterion(pred_pos, torch.ones_like(pred_pos)) + creterion(pred_neg, torch.zeros_like(pred_neg))
            if q_target and q_context:
                # Take mean over batch and sum over
                kl_loss = kl_divergence(q_target, q_context).sum(dim=1).mean()
                total_kl_loss += kl_loss.item()
                loss = nll + kl_loss
            else:
                loss = nll
        total_loss += loss.item()
        total_nll_loss += nll.item()
        loss.backward()
        optimizer.step()
        model.detach()  # Detach for memory optimization
        t_prep_s = time.time()
        if mailbox is not None:
            eid = rows['Unnamed: 0'].values
            mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
            block = None
            if memory_param['deliver_to'] == 'neighbors':
                block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
            mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block, neg_samples=train_neg_samples)
            mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts, neg_samples=train_neg_samples)
        time_prep += time.time() - t_prep_s
        time_tot += time.time() - t_tot_s
    ap, auc, auc_mrr = eval('val')
    if ap > best_ap:
        best_e = e
        best_ap = ap
        torch.save(model.state_dict(), path_saver)
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"EarlyStopping counter: {early_stop_counter} out of {train_param['patience']}")
        if early_stop_counter > train_param['patience']:
            break
    print(
        '\ttrain loss:{:.4f}, train nll:{:.4f}, train kl loss:{:.5f}  val ap:{:4f}  val auc:{:4f} val mrr:{:.4f}'.format(
            total_loss / batch_num, total_nll_loss / batch_num, total_kl_loss / batch_num,
            ap, auc, auc_mrr))
    print('\ttotal time:{:.2f}s sample time:{:.2f}s prep time:{:.2f}s'.format(time_tot, time_sample, time_prep))

print('Loading model at epoch {}...'.format(best_e))
model.load_state_dict(torch.load(path_saver))
model.eval()
print("Testing...")
if sampler is not None:
    sampler.reset()
if mailbox is not None:
    mailbox.reset()
    model.memory_updater.last_updated_nid = None
model.reset()
model.test = True
if args.base_model != 'origin' or mailbox is not None:
    eval('train', stage='test')
    eval('val', stage='test')
all_ap, all_auc, all_mrr, raw_results = eval('test')
results_by_time = eval_by_ts(raw_results, split_num=train_param.get('test_split_num', 4))
print('\t Overall test ap:{:4f}  Overall test auc:{:4f}  Overall test mrr:{:4f}'.format(all_ap, all_auc, all_mrr))
for s in results_by_time:
    print(f"{s[0]}: ap: {s[1]['ap']}, auc: {s[1]['auc']}, mrr: {s[1]['mrr']}, loss: {s[1]['loss']}")
