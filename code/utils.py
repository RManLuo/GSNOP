import torch
import os
import yaml
import dgl
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss


def load_feat(d, n_node=0, n_edge=0, rand_de=0, rand_dn=0, zero_feat=False):
    node_feats = None
    if os.path.exists('DATA/{}/node_features.pt'.format(d)):
        node_feats = torch.load('DATA/{}/node_features.pt'.format(d))
        if node_feats.dtype == torch.bool:
            node_feats = node_feats.type(torch.float32)
    edge_feats = None
    if os.path.exists('DATA/{}/edge_features.pt'.format(d)):
        edge_feats = torch.load('DATA/{}/edge_features.pt'.format(d))
        if edge_feats.dtype == torch.bool:
            edge_feats = edge_feats.type(torch.float32)
    if rand_de > 0:
        assert n_edge > 0
        edge_feats = torch.randn(n_edge, rand_de)
    if rand_dn > 0:
        assert n_node > 0
        node_feats = torch.randn(n_node, rand_dn)
    if zero_feat:
        if node_feats is not None:
            node_feats = torch.zeros_like(node_feats)
        if edge_feats is not None:
            edge_feats = torch.zeros_like(edge_feats)
    return node_feats, edge_feats


def load_graph(d):
    df = pd.read_csv('DATA/{}/edges.csv'.format(d))
    g = np.load('DATA/{}/ext_full.npz'.format(d))
    return g, df


def parse_config(f):
    conf = yaml.safe_load(open(f, 'r'))
    sample_param = conf['sampling'][0]
    memory_param = conf['memory'][0]
    gnn_param = conf['gnn'][0]
    np_param = conf['np'][0]
    train_param = conf['train'][0]
    return sample_param, memory_param, gnn_param, np_param, train_param


def to_dgl_blocks(ret, hist, reverse=False, cuda=True):
    mfgs = list()
    for r in ret:
        if not reverse:
            b = dgl.create_block(
                (r.col(), r.row()), num_src_nodes=r.dim_in(), num_dst_nodes=r.dim_out())
            b.srcdata['ID'] = torch.from_numpy(r.nodes())
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_dst_nodes():]
            b.srcdata['ts'] = torch.from_numpy(r.ts())
        else:
            b = dgl.create_block(
                (r.row(), r.col()), num_src_nodes=r.dim_out(), num_dst_nodes=r.dim_in())
            b.dstdata['ID'] = torch.from_numpy(r.nodes())
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_src_nodes():]
            b.dstdata['ts'] = torch.from_numpy(r.ts())
        b.edata['ID'] = torch.from_numpy(r.eid())
        if cuda:
            mfgs.append(b.to('cuda:0'))
        else:
            mfgs.append(b)
    mfgs = list(map(list, zip(*[iter(mfgs)] * hist)))
    mfgs.reverse()
    return mfgs


def node_to_dgl_blocks(root_nodes, ts, cuda=True):
    mfgs = list()
    b = dgl.create_block(
        ([], []), num_src_nodes=root_nodes.shape[0], num_dst_nodes=root_nodes.shape[0])
    b.srcdata['ID'] = torch.from_numpy(root_nodes)
    b.srcdata['ts'] = torch.from_numpy(ts)
    if cuda:
        mfgs.insert(0, [b.to('cuda:0')])
    else:
        mfgs.insert(0, [b])
    return mfgs


def mfgs_to_cuda(mfgs):
    for mfg in mfgs:
        for i in range(len(mfg)):
            mfg[i] = mfg[i].to('cuda:0')
    return mfgs


def prepare_input(mfgs, node_feats, edge_feats, combine_first=False, pinned=False, nfeat_buffs=None, efeat_buffs=None,
                  nids=None, eids=None):
    if combine_first:
        for i in range(len(mfgs[0])):
            if mfgs[0][i].num_src_nodes() > mfgs[0][i].num_dst_nodes():
                num_dst = mfgs[0][i].num_dst_nodes()
                ts = mfgs[0][i].srcdata['ts'][num_dst:]
                nid = mfgs[0][i].srcdata['ID'][num_dst:].float()
                nts = torch.stack([ts, nid], dim=1)
                unts, idx = torch.unique(nts, dim=0, return_inverse=True)
                uts = unts[:, 0]
                unid = unts[:, 1]
                # import pdb; pdb.set_trace()
                b = dgl.create_block((idx + num_dst, mfgs[0][i].edges()[1]), num_src_nodes=unts.shape[0] + num_dst,
                                     num_dst_nodes=num_dst, device=torch.device('cuda:0'))
                b.srcdata['ts'] = torch.cat(
                    [mfgs[0][i].srcdata['ts'][:num_dst], uts], dim=0)
                b.srcdata['ID'] = torch.cat(
                    [mfgs[0][i].srcdata['ID'][:num_dst], unid], dim=0)
                b.edata['dt'] = mfgs[0][i].edata['dt']
                b.edata['ID'] = mfgs[0][i].edata['ID']
                mfgs[0][i] = b
    t_idx = 0
    t_cuda = 0
    i = 0
    if node_feats is not None:
        for b in mfgs[0]:
            if pinned:
                if nids is not None:
                    idx = nids[i]
                else:
                    idx = b.srcdata['ID'].cpu().long()
                torch.index_select(node_feats, 0, idx,
                                   out=nfeat_buffs[i][:idx.shape[0]])
                b.srcdata['h'] = nfeat_buffs[i][:idx.shape[0]].cuda(
                    non_blocking=True)
                i += 1
            else:
                srch = node_feats[b.srcdata['ID'].long()].float()
                b.srcdata['h'] = srch.cuda()
    i = 0
    if edge_feats is not None:
        for mfg in mfgs:
            for b in mfg:
                if b.num_src_nodes() > b.num_dst_nodes():
                    if pinned:
                        if eids is not None:
                            idx = eids[i]
                        else:
                            idx = b.edata['ID'].cpu().long()
                        torch.index_select(
                            edge_feats, 0, idx, out=efeat_buffs[i][:idx.shape[0]])
                        b.edata['f'] = efeat_buffs[i][:idx.shape[0]].cuda(
                            non_blocking=True)
                        i += 1
                    else:
                        srch = edge_feats[b.edata['ID'].long()].float()
                        b.edata['f'] = srch.cuda()
    return mfgs


def get_ids(mfgs, node_feats, edge_feats):
    nids = list()
    eids = list()
    if node_feats is not None:
        for b in mfgs[0]:
            nids.append(b.srcdata['ID'].long())
    if edge_feats is not None:
        for mfg in mfgs:
            for b in mfg:
                eids.append(b.edata['ID'].long())
    return nids, eids


def get_pinned_buffers(sample_param, batch_size, node_feats, edge_feats):
    pinned_nfeat_buffs = list()
    pinned_efeat_buffs = list()
    limit = int(batch_size * 3.3)
    if 'neighbor' in sample_param:
        for i in sample_param['neighbor']:
            limit *= i + 1
            if edge_feats is not None:
                for _ in range(sample_param['history']):
                    pinned_efeat_buffs.insert(0, torch.zeros(
                        (limit, edge_feats.shape[1]), pin_memory=True))
    if node_feats is not None:
        for _ in range(sample_param['history']):
            pinned_nfeat_buffs.insert(0, torch.zeros(
                (limit, node_feats.shape[1]), pin_memory=True))
    return pinned_nfeat_buffs, pinned_efeat_buffs


class ContextTargetSpliter():
    def __init__(self, context_split):
        self.context_split = context_split

    def __call__(self, data):
        c_src, t_src, c_pos_dst, t_pos_dst, c_neg_dst, t_neg_dst, c_ts_src, t_ts_src, c_ts_pos_dst, t_ts_pos_dst, c_ts_neg_dst, t_ts_neg_dst = train_test_split(
            *data,
            train_size=self.context_split)
        return [c_src, c_pos_dst, c_neg_dst], [t_src, t_pos_dst, t_neg_dst], [c_ts_src, c_ts_pos_dst, c_ts_neg_dst], [
            t_ts_src, t_ts_pos_dst, t_ts_neg_dst]


def eval_by_ts(results, split_num=5):
    results, mrr_results = results
    max_t = results['t'].max()
    min_t = results['t'].min()
    step = 1.0 / split_num
    result_by_time = []
    start_t = min_t
    start_ratio = 0
    split_time = [(max_t-min_t) * step * (s + 1) + min_t for s in range(split_num)]
    for i, t in enumerate(split_time):
        ratio = step * (i + 1)
        d = results[(results['t'] >= start_t) & (results['t'] < t)]
        mrr = -1
        if mrr_results is not None:
            mrr_d = mrr_results[(mrr_results['t'] >= start_t) & (mrr_results['t'] < t)]
            if len(d) !=0:
                mrr = mrr_d['mrr'].mean()
        if len(d) ==0:
            ap = -1
            auc = -1
            loss = -1
        else:
            ap = average_precision_score(d['y_true'], d['y_pred'])
            auc = roc_auc_score(d['y_true'], d['y_pred'])
            loss = log_loss(d['y_true'], d['y_pred'], eps=1e-8)
        result_by_time.append(
            (f"{start_ratio:.2f}-{ratio:.2f}", {'ap': ap, "auc": auc, 'mrr': mrr,'loss': loss}))
        start_t = t
        start_ratio = ratio
    return result_by_time
