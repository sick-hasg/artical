# #!/usr/bin/env python
# import os
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#
# import click as ck
# import pandas as pd
# from Bio import SeqIO
# from toolkit.extract_esm import extract_esm
#
#
# @ck.command()
# @ck.option('--data-root', '-dr', default='cafa_data', required=True, help='CAFA5 data root')
# @ck.option('--esm-root', '-er', default='data', required=True, help='esm2 model file root')
# @ck.option('--file-name', '-fn', required=True, help='proteins file name')
# @ck.option('--device', '-d', default='cuda:0')
# def main(data_root, esm_root, file_name, device):
#     for ont in ['mf']:
#         print(f'Processing {ont} proteins.....')
#     # 加载fasta和pkl数据
#
#         fasta_file = f'{data_root}/{ont}/{file_name}.fasta'
#         pkl_file = f'{data_root}/{ont}/{file_name}.pkl'
#         out_file = f'{data_root}/{ont}/{file_name}_esmEmbeddings.pkl'
#         df = pd.read_pickle(pkl_file)
#         print(f"Loaded {len(df)} proteins from {pkl_file}")
#
#         # Optional: Extract ESM2 embeddings (same as your original script)
#         print('Extracting ESM2 embeddings...')
#         prots, esm2_data = extract_esm(fasta_file, esm_root, device=device)
#         esm_list = [emb for emb in esm2_data]  # List of 2560-dim tensors
#
#         print('Build mapping: protein_id -> embedding')
#         emb_dict = dict(zip(prots, esm_list))
#
#         # Align with df['proteins'] order
#         aligned_embeddings = []
#         for pid in df['proteins']:
#             aligned_embeddings.append(emb_dict[pid])
#
#         # Add new column 'esm2'
#         df['esm2'] = aligned_embeddings
#
#         # Save updated DataFrame
#         df.to_pickle(out_file)
#         print(f'Saved {len(df)} proteins with ESM2 embeddings to {out_file}')
#
#
# if __name__ == '__main__':
#     main()


#谢红，2025.11.9重新写主调函数
#!/usr/bin/env python
import os
import sys
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import click as ck
import pandas as pd
from toolkit.extract_esm import extract_esm


@ck.command()
@ck.option('--data-root', '-dr', default='cafa_data', type=ck.Path(exists=True), help='CAFA5 data root directory')
@ck.option('--esm-root', '-er', default='data', type=ck.Path(exists=True), help='Directory containing ESM2 model .pt files')
@ck.option('--file-name', '-fn', required=True, help='Base name of protein files (e.g., "train", "test")')
@ck.option('--ontologies', '-ont', multiple=True, default=['mf', 'bp', 'cc'],
            type=ck.Choice(['mf', 'bp', 'cc']), help='Ontologies to process (can specify multiple)')
@ck.option('--device', '-d', default='auto',
           help="Device to use: 'cuda:X', 'cpu', or 'auto' (default: auto -> GPU if available)")

def main(data_root, esm_root, file_name, ontologies, device):
    # 自动选择设备
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    for ont in ontologies:
        print(f'\n{"="*50}')
        print(f'Processing ontology: {ont.upper()}')
        print(f'{"="*50}')

        # 文件路径
        fasta_file = os.path.join(data_root, ont, f'{file_name}.fasta')
        pkl_file   = os.path.join(data_root, ont, f'{file_name}.pkl')
        out_file   = os.path.join(data_root, ont, f'{file_name}_esmEmbeddings.pkl')
        cache_file = os.path.join(data_root, ont, f'{file_name}_esm2_raw.pt')  # 原始嵌入缓存

        # 检查输入文件是否存在
        if not os.path.exists(fasta_file):
            print(f"⚠️  FASTA file not found: {fasta_file}. Skipping {ont}.")
            continue
        if not os.path.exists(pkl_file):
            print(f"⚠️  PKL file not found: {pkl_file}. Skipping {ont}.")
            continue

        # 加载原始 DataFrame
        df = pd.read_pickle(pkl_file)
        print(f"✅ Loaded {len(df)} proteins from {pkl_file}")

        # 提取 ESM 嵌入（带缓存）
        print("🔍 Extracting ESM2 embeddings (cached if exists)...")
        prots, esm2_data = extract_esm(
            fasta_file=fasta_file,
            base_location=esm_root,
            device=device,
            out_file=cache_file,  # 启用缓存！
            truncation_seq_length=1022,
            toks_per_batch=4096
        )
        print(f"✅ Got {len(prots)} embeddings from ESM2.")

        # 构建映射
        emb_dict = dict(zip(prots, [emb.clone() for emb in esm2_data]))

        # === 关键校验：ID 是否完全匹配 ===
        df_protein_set = set(df['proteins'])
        fasta_protein_set = set(prots)
        if df_protein_set != fasta_protein_set:
            missing_in_fasta = df_protein_set - fasta_protein_set
            missing_in_df = fasta_protein_set - df_protein_set
            print(f"❌ ID mismatch between FASTA and DataFrame in {ont}:")
            if missing_in_fasta:
                print(f"  - In DataFrame but not in FASTA ({len(missing_in_fasta)}): {list(missing_in_fasta)[:3]}...")
            if missing_in_df:
                print(f"  - In FASTA but not in DataFrame ({len(missing_in_df)}): {list(missing_in_df)[:3]}...")
            raise ValueError("Protein ID sets do not match. Please check your data alignment.")

        # 对齐嵌入顺序
        print("🔁 Aligning embeddings with DataFrame order...")
        aligned_embeddings = [emb_dict[pid] for pid in df['proteins']]

        # 添加新列
        df['esm2'] = aligned_embeddings

        # 保存结果
        df.to_pickle(out_file)
        print(f"✅ Saved updated DataFrame with ESM2 embeddings to: {out_file}")


if __name__ == '__main__':
    main()