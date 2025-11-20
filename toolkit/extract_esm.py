#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from pathlib import Path
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained
import os
import gzip

class GzippedFastaBatchedDataset(FastaBatchedDataset):

    @classmethod
    def from_file(cls, fasta_file):
        sequence_labels, sequence_strs = [], []
        cur_seq_label = None
        buf = []

        def _flush_current_seq():
            nonlocal cur_seq_label, buf
            if cur_seq_label is None:
                return
            sequence_labels.append(cur_seq_label)
            sequence_strs.append("".join(buf))
            cur_seq_label = None
            buf = []

        with gzip.open(fasta_file, "rt") as infile:
            for line_idx, line in enumerate(infile):
                if line.startswith(">"):  # label line
                    _flush_current_seq()
                    line = line[1:].strip()
                    if len(line) > 0:
                        cur_seq_label = line
                    else:
                        cur_seq_label = f"seqnum{line_idx:09d}"
                else:  # sequence line
                    buf.append(line.strip())

        _flush_current_seq()

        assert len(set(sequence_labels)) == len(
            sequence_labels
        ), "Found duplicate sequence labels"

        return cls(sequence_labels, sequence_strs)


# def extract_esm(fasta_file, base_location,
#                 truncation_seq_length=1022, toks_per_batch=4096,
#                 device=None, out_file=None):
#     if out_file is not None and os.path.exists(out_file):
#         obj = torch.load(out_file)
#         data = obj['data']
#         proteins = obj['proteins']
#         return proteins, data
#     model_location = f'{base_location}/esm2_t36_3B_UR50D.pt'
#     model, alphabet = pretrained.load_model_and_alphabet(model_location)
#     model.eval()
#     if device:
#         model = model.to(device)
#
#     #torch.load操作导致模型重新加载，2025.11.9，xiehong 删除后续两行
#     #map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     #model_data = torch.load(str(model_location), map_location=map_location)
#
#     if fasta_file.endswith('.gz'):
#         dataset = GzippedFastaBatchedDataset.from_file(fasta_file)
#     else:
#         dataset = FastaBatchedDataset.from_file(fasta_file)
#     batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
#     data_loader = torch.utils.data.DataLoader(
#         dataset, collate_fn=alphabet.get_batch_converter(truncation_seq_length), batch_sampler=batches
#     )
#     print(f"Read {fasta_file} with {len(dataset)} sequences")
#
#     return_contacts = False
#
#     repr_layers = [36,]
#
#     proteins = []
#     data = []
#     with torch.no_grad():
#         for batch_idx, (labels, strs, toks) in enumerate(data_loader):
#             print(
#                 f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
#             )
#             if device:
#                 toks = toks.to(device, non_blocking=True)
#
#             out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)
#
#             logits = out["logits"].to(device="cpu")
#             representations = {
#                 layer: t.to(device="cpu") for layer, t in out["representations"].items()
#             }
#             if return_contacts:
#                 contacts = out["contacts"].to(device="cpu")
#
#             for i, label in enumerate(labels):
#                 result = {"label": label}
#                 truncate_len = min(truncation_seq_length, len(strs[i]))
#                 result["mean_representations"] = {
#                     layer: t[i, 1 : truncate_len + 1].mean(0).clone()
#                     for layer, t in representations.items()
#                 }
#                 proteins.append(label)
#                 data.append(result["mean_representations"][36])
#     data = torch.stack(data).reshape(-1, 2560)
#     if out_file is not None:
#         torch.save({'data': data, 'proteins': proteins}, out_file)
#     return proteins, data


def extract_esm(
    fasta_file,
    base_location,
    truncation_seq_length=1022,
    toks_per_batch=4096,
    device=None,
    out_file=None,
    model_name="esm2_t36_3B_UR50D",  # 可扩展：支持 esm2_t12_35M_UR50D 等
):
    """
    Extract mean-pooled ESM2 embeddings for protein sequences in a FASTA file.

    Args:
        fasta_file (str): Path to FASTA file (plain text, .gz not supported here).
        base_location (str): Directory containing ESM2 .pt model files.
        truncation_seq_length (int): Max sequence length (default: 1022).
        toks_per_batch (int): Max tokens per batch for memory efficiency.
        device (str or torch.device): Device to run inference on.
        out_file (str, optional): Cache file path (.pt). If exists, load from it.
        model_name (str): ESM2 model name (e.g., 'esm2_t36_3B_UR50D').

    Returns:
        proteins (List[str]): Protein IDs from FASTA headers.
        data (torch.Tensor): Embeddings of shape (N, D), where D depends on model.
    """
    # ====== 1. Load from cache if available ======
    if out_file is not None and os.path.exists(out_file):
        print(f"Loading cached ESM embeddings from {out_file}")
        obj = torch.load(out_file, map_location="cpu")
        return obj["proteins"], obj["data"]

    # ====== 2. Load ESM2 model ======
    model_file = f"{model_name}.pt"
    model_location = os.path.join(base_location, model_file)
    if not os.path.exists(model_location):
        raise FileNotFoundError(f"ESM model not found at {model_location}")

    print(f"Loading ESM model: {model_file}")
    model, alphabet = pretrained.load_model_and_alphabet(model_location)
    model.eval()

    if device:
        model = model.to(device)

    # ====== 3. Prepare dataset ======
    if fasta_file.endswith('.gz'):
        raise ValueError(
            "Gzipped FASTA is not supported in this version. "
            "Please decompress first (e.g., gunzip file.fasta.gz)."
        )

    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    batch_converter = alphabet.get_batch_converter(truncation_seq_length)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=batch_converter, batch_sampler=batches
    )
    print(f"Read {len(dataset)} sequences from {fasta_file}")

    # ====== 4. Determine representation layer and output dimension ======
    # Infer layer and dimension from model_name (extendable)
    if "t36" in model_name:
        repr_layer = 36
        embed_dim = 2560
    elif "t33" in model_name:
        repr_layer = 33
        embed_dim = 1280
    elif "t30" in model_name:
        repr_layer = 30
        embed_dim = 1280
    elif "t12" in model_name:
        repr_layer = 12
        embed_dim = 480
    else:
        raise ValueError(f"Unsupported model: {model_name}. Please specify layer/dim manually.")

    proteins = []
    embeddings_list = []

    # ====== 5. Inference ======
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f"Processing batch {batch_idx + 1}/{len(batches)} ({toks.size(0)} seqs)")
            if device:
                toks = toks.to(device, non_blocking=True)

            # Only compute necessary representations
            out = model(toks, repr_layers=[repr_layer], return_contacts=False)

            # Move representations to CPU immediately to save GPU memory
            reps = out["representations"][repr_layer].to(device="cpu")

            for i, label in enumerate(labels):
                seq_len = len(strs[i])
                truncate_len = min(truncation_seq_length, seq_len)
                # Skip [CLS] token at position 0; take 1 : truncate_len + 1
                seq_rep = reps[i, 1: truncate_len + 1].mean(dim=0)  # (D,)
                proteins.append(label)
                embeddings_list.append(seq_rep.clone())

            # Optional: clear GPU cache (if needed for very large datasets)
            # torch.cuda.empty_cache()

    # ====== 6. Finalize and save ======
    data = torch.stack(embeddings_list)  # Shape: (N, D)
    assert data.shape == (len(proteins), embed_dim), f"Expected ({len(proteins)}, {embed_dim}), got {data.shape}"

    if out_file is not None:
        torch.save({"proteins": proteins, "data": data}, out_file)
        print(f"Saved ESM embeddings to {out_file}")

    return proteins, data




