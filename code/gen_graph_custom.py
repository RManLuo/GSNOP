import argparse
import itertools
import os.path

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from sklearn.preprocessing import LabelEncoder
import math

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--raw_data', type=str, help='raw data name')
parser.add_argument('--add_reverse', default=False, action='store_true')
parser.add_argument('--val_ratio', default=0.70, type=float)
parser.add_argument('--test_ratio', default=0.85, type=float)
args = parser.parse_args()

# Processing RAW data
raw_df = pd.read_csv('DATA/{}/{}.csv'.format(args.data, args.raw_data), index_col=0,
                     names=['src', 'dst', 'time', 'int_roll', 'ext_roll'], header=0)
raw_df['time'] = raw_df['time'] - raw_df['time'].min()
max_t = raw_df['time'].max()
min_t = raw_df['time'].min()
# val_time, test_time = list(np.quantile(raw_df.time, [args.val_ratio, args.test_ratio]))
val_time = (max_t - min_t ) * args.val_ratio + min_t
test_time = (max_t - min_t ) * args.test_ratio + min_t
raw_df.loc[raw_df['time'] < val_time, 'ext_roll'] = 0
raw_df.loc[(raw_df['time'] >= val_time) & (raw_df['time'] <= test_time), 'ext_roll'] = 1
raw_df.loc[raw_df['time'] > test_time, "ext_roll"] = 2
raw_df.to_csv('DATA/{}/edges.csv'.format(args.data))
# Load features
if os.path.exists('DATA/{}/{}_node.npy'.format(args.data, args.raw_data)):
    node_features = torch.from_numpy(np.load('DATA/{}/{}_node.npy'.format(args.data, args.raw_data))).float()
    torch.save(node_features, 'DATA/{}/node_features.pt'.format(args.data))
if os.path.exists('DATA/{}/{}.npy'.format(args.data, args.raw_data)):
    edge_features = torch.from_numpy(np.load('DATA/{}/{}.npy'.format(args.data, args.raw_data))).float()
    torch.save(edge_features, 'DATA/{}/edge_features.pt'.format(args.data))

df = pd.read_csv('DATA/{}/edges.csv'.format(args.data))
num_nodes = max(int(df['src'].max()), int(df['dst'].max())) + 1
print('num_nodes: ', num_nodes)

int_train_indptr = np.zeros(num_nodes + 1, dtype=np.int32)
int_train_indices = [[] for _ in range(num_nodes)]
int_train_ts = [[] for _ in range(num_nodes)]
int_train_eid = [[] for _ in range(num_nodes)]

int_full_indptr = np.zeros(num_nodes + 1, dtype=np.int32)
int_full_indices = [[] for _ in range(num_nodes)]
int_full_ts = [[] for _ in range(num_nodes)]
int_full_eid = [[] for _ in range(num_nodes)]

ext_full_indptr = np.zeros(num_nodes + 1, dtype=np.int32)
ext_full_indices = [[] for _ in range(num_nodes)]
ext_full_ts = [[] for _ in range(num_nodes)]
ext_full_eid = [[] for _ in range(num_nodes)]

for idx, row in tqdm(df.iterrows(), total=len(df)):
    src = int(row['src'])
    dst = int(row['dst'])
    if row['int_roll'] == 0:
        int_train_indices[src].append(dst)
        int_train_ts[src].append(row['time'])
        int_train_eid[src].append(idx)
        if args.add_reverse:
            int_train_indices[dst].append(src)
            int_train_ts[dst].append(row['time'])
            int_train_eid[dst].append(idx)
        # int_train_indptr[src + 1:] += 1
    if row['int_roll'] != 3:
        int_full_indices[src].append(dst)
        int_full_ts[src].append(row['time'])
        int_full_eid[src].append(idx)
        if args.add_reverse:
            int_full_indices[dst].append(src)
            int_full_ts[dst].append(row['time'])
            int_full_eid[dst].append(idx)
        # int_full_indptr[src + 1:] += 1
    ext_full_indices[src].append(dst)
    ext_full_ts[src].append(row['time'])
    ext_full_eid[src].append(idx)
    if args.add_reverse:
        ext_full_indices[dst].append(src)
        ext_full_ts[dst].append(row['time'])
        ext_full_eid[dst].append(idx)
    # ext_full_indptr[src + 1:] += 1

for i in tqdm(range(num_nodes)):
    int_train_indptr[i + 1] = int_train_indptr[i] + len(int_train_indices[i])
    int_full_indptr[i + 1] = int_full_indptr[i] + len(int_full_indices[i])
    ext_full_indptr[i + 1] = ext_full_indptr[i] + len(ext_full_indices[i])

int_train_indices = np.array(list(itertools.chain(*int_train_indices)))
int_train_ts = np.array(list(itertools.chain(*int_train_ts)))
int_train_eid = np.array(list(itertools.chain(*int_train_eid)))

int_full_indices = np.array(list(itertools.chain(*int_full_indices)))
int_full_ts = np.array(list(itertools.chain(*int_full_ts)))
int_full_eid = np.array(list(itertools.chain(*int_full_eid)))

ext_full_indices = np.array(list(itertools.chain(*ext_full_indices)))
ext_full_ts = np.array(list(itertools.chain(*ext_full_ts)))
ext_full_eid = np.array(list(itertools.chain(*ext_full_eid)))

print('Sorting...')


def tsort(i, indptr, indices, t, eid):
    beg = indptr[i]
    end = indptr[i + 1]
    sidx = np.argsort(t[beg:end])
    indices[beg:end] = indices[beg:end][sidx]
    t[beg:end] = t[beg:end][sidx]
    eid[beg:end] = eid[beg:end][sidx]


for i in tqdm(range(int_train_indptr.shape[0] - 1)):
    tsort(i, int_train_indptr, int_train_indices, int_train_ts, int_train_eid)
    tsort(i, int_full_indptr, int_full_indices, int_full_ts, int_full_eid)
    tsort(i, ext_full_indptr, ext_full_indices, ext_full_ts, ext_full_eid)

# import pdb; pdb.set_trace()
print('saving...')
np.savez('DATA/{}/int_train.npz'.format(args.data), indptr=int_train_indptr, indices=int_train_indices, ts=int_train_ts,
         eid=int_train_eid)
np.savez('DATA/{}/int_full.npz'.format(args.data), indptr=int_full_indptr, indices=int_full_indices, ts=int_full_ts,
         eid=int_full_eid)
np.savez('DATA/{}/ext_full.npz'.format(args.data), indptr=ext_full_indptr, indices=ext_full_indices, ts=ext_full_ts,
         eid=ext_full_eid)
