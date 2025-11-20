#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import math
import gzip
from toolkit.utils import Ontology
from toolkit.models import GINModel
from toolkit.extract_esm import extract_esm
import torch as th
import os
import dgl
from toolkit.data import load_ppi_data
from tqdm import tqdm


@ck.command()
@ck.option('--in-file', '-if', default='example.fa', help='Input FASTA file', required=True)
@ck.option(
    '--data-root', '-dr', default='data',
    help='Prediction model root')
@ck.option('--threshold', '-t', default=0.8, help='Prediction threshold')
@ck.option('--batch-size', '-bs', default=32, help='Batch size for prediction model')
@ck.option(
    '--device', '-d', default='cpu',
    help='Device')
@ck.option('--model-name', '-mn', default='magingo', help='Model name')
@ck.option(
    '--test-data-name', '-td', default='test', type=ck.Choice(['test', 'nextprot', 'valid']),
    help='Test data set name')
def main(in_file, data_root, threshold, batch_size, device, model_name, test_data_name):
    fn = os.path.splitext(in_file)[0]
    out_file_esm = f'{fn}_esm.pkl'
    proteins, data = extract_esm(in_file, data_root, out_file=out_file_esm, device=device)

    go_file = f'{data_root}/go.obo'
    go_norm = f'{data_root}/go-plus.norm'
    go = Ontology(go_file, with_rels=True)

    for ont in ['mf', 'cc', 'bp']:
        print(f'Predicting {ont} classes')
        terms_file = f'{data_root}/{ont}/{ont}o_terms.pkl'
        out_file = f'{fn}_preds_{ont}.tsv.gz'
        terms_df = pd.read_pickle(terms_file)
        terms = terms_df['gos'].values.flatten()
        terms_dict = {v: i for i, v in enumerate(terms)}

        n_terms = len(terms_dict)
        model = GINModel(2560, n_terms, device).to(device)
        model_file = f'{data_root}/{ont}/{model_name}.th'
        model.load_state_dict(th.load(model_file, map_location=device))
        model.eval()

        features_length = 2560
        features_column = "esm2"
        test_data_file = f'{test_data_name}_data.pkl'
        ppi_graph_file = f'ppi_{test_data_name}.bin'
        mfs_dict, terms_dict, graph, train_nids, valid_nids, test_nids, data, labels, test_df, go_embedding, all_nids = load_ppi_data(
            data_root, ont, features_length, features_column, test_data_file, ppi_graph_file)
        go_embedding = go_embedding.mean(dim=0, keepdim=True).to(device)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        dataloader = dgl.dataloading.DataLoader(
            graph, all_nids, sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0)

        with th.no_grad():
            preds = []
            steps = int(math.ceil(len(all_nids) / batch_size))
            with tqdm(total=steps) as bar:
                for input_nodes, output_nodes, blocks in dataloader:
                    bar.update(1)
                    logits = model(input_nodes, output_nodes, blocks, None, go_embedding)
                    if isinstance(logits, tuple):
                        _, graph_logits = logits
                        batch_preds = graph_logits
                    else:
                        batch_preds = logits
                    preds.append(batch_preds.detach().cpu().numpy())

            preds = np.concatenate(preds)

        with gzip.open(out_file, 'wt') as f:
            for i in range(len(proteins)):
                above_threshold = np.argwhere(preds[i] >= threshold).flatten()
                for j in above_threshold:
                    name = go.get_term(terms[j])['name']
                    f.write(f'{proteins[i]}\t{terms[j]}\t{preds[i, j]:0.3f}\n')

if __name__ == '__main__':
    main()