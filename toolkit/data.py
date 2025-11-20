import pandas as pd
import torch as th
import numpy as np
import dgl
import os


def get_data(df, features_dict, terms_dict, features_length, features_column):
    """
    Converts dataframe file with protein information and returns
    PyTorch tensors
    """
    data = th.zeros((len(df), features_length), dtype=th.float32)
    labels = th.zeros((len(df), len(terms_dict)), dtype=th.float32)
    for i, row in enumerate(df.itertuples()):
        # Data vector
        if features_column == 'esm2':
            data[i, :] = th.FloatTensor(row.esm2)
        elif features_column == 'interpros':
            for feat in row.interpros:
                if feat in features_dict:
                    data[i, features_dict[feat]]
        elif features_column == 'mf_preds':
            data[i, :] = th.FloatTensor(row.mf_preds)
        elif features_column == 'prop_annotations':
            for feat in row.prop_annotations:
                if feat in features_dict:
                    data[i, features_dict[feat]] = 1
        # Labels vector
        for go_id in row.prop_annotations:
            if go_id in terms_dict:
                g_id = terms_dict[go_id]
                labels[i, g_id] = 1
    return data, labels

def load_data(
        data_root, ont, terms_file, features_length=2560,
        features_column='esm2', test_data_file='test_data.pkl'):
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    print('Terms', len(terms))
    
    ipr_df = pd.read_pickle(f'{data_root}/{ont}/interpros.pkl')
    iprs = ipr_df['interpros'].values
    iprs_dict = {v:k for k, v in enumerate(iprs)}
    if features_column == 'interpros':
        features_length = len(iprs_dict)
    

    train_df = pd.read_pickle(f'{data_root}/{ont}/train_data.pkl')
    valid_df = pd.read_pickle(f'{data_root}/{ont}/valid_data.pkl')
    test_df = pd.read_pickle(f'{data_root}/{ont}/{test_data_file}')

    train_data = get_data(train_df, iprs_dict, terms_dict, features_length, features_column)
    valid_data = get_data(valid_df, iprs_dict, terms_dict, features_length, features_column)
    test_data = get_data(test_df, iprs_dict, terms_dict, features_length, features_column)

    return iprs_dict, terms_dict, train_data, valid_data, test_data, test_df


def load_ppi_data(data_root, ont, features_length=2560,
                  features_column='esm2', test_data_file='test_data.pkl',
                  ppi_graph_file='ppi_test.bin'):
    terms_df = pd.read_pickle(f'{data_root}/{ont}/terms.pkl')
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    print('Terms', len(terms))
    anc2vec_dict = np.load(f'{data_root}/label-embedding-128_deepSE.npy', allow_pickle=True)
    go_emb_matrix = np.zeros((len(terms), 128))
    es = anc2vec_dict.item()
    go_emb = []
    noexist_go_emb = []
    for i in range(len(terms)):
        if es.get(terms[i]) is not None:
            go_embedding = es[terms[i]]
            go_emb.append(go_embedding)
            go_emb_matrix[i] = es[terms[i]]
        else:
            noexist_go_emb.append(terms[i])
    go_emb_tensor = th.from_numpy(go_emb_matrix).float()

    mf_df = pd.read_pickle(f'{data_root}/mf/terms.pkl')
    mfs = mf_df['gos'].values
    mfs_dict = {v: k for k, v in enumerate(mfs)}

    ipr_df = pd.read_pickle(f'{data_root}/{ont}/interpros.pkl')
    iprs = ipr_df['interpros'].values
    iprs_dict = {v: k for k, v in enumerate(iprs)}

    feat_dict = None

    if features_column == 'interpros':
        features_length = len(iprs_dict)
        feat_dict = iprs_dict
    elif features_column != 'esm2':
        features_length = len(mfs_dict)
        feat_dict = mfs_dict

    train_df = pd.read_pickle(f'{data_root}/{ont}/train_data.pkl')
    valid_df = pd.read_pickle(f'{data_root}/{ont}/valid_data.pkl')
    test_df = pd.read_pickle(f'{data_root}/{ont}/{test_data_file}')

    df = pd.concat([train_df, valid_df, test_df])
    graphs, nids = dgl.load_graphs(f'{data_root}/{ont}/{ppi_graph_file}')
    data, labels = get_data(df, feat_dict, terms_dict, features_length, features_column)
    graph = graphs[0]
    graph.ndata['feat'] = data
    graph.ndata['labels'] = labels
    train_nids, valid_nids, test_nids = nids['train_nids'], nids['valid_nids'], nids['test_nids']
    all_nids = th.cat([train_nids, valid_nids, test_nids])
    return feat_dict, terms_dict, graph, train_nids, valid_nids, test_nids, data, labels, test_df, go_emb_tensor, all_nids