#!/usr/bin/env python3
import argparse
import sys
import click as ck


def extract_uniprot_ids(fasta_file):
    """
    从 FASTA 文件中提取 UniProt ID（Accession）
    支持格式: >sp|P12345|... 或 >tr|Q9Y6K9|...
    """
    uniprot_ids = []
    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                # 按 | 分割，取中间部分（索引 1）
                parts = line[1:].strip().split('|')
                if len(parts) >= 2:
                    uniprot_id = parts[1]
                    # 可选：验证是否为有效 UniProt ID（字母+数字，5-10位）
                    if uniprot_id and any(c.isalpha() for c in uniprot_id) and any(c.isdigit() for c in uniprot_id):
                        uniprot_ids.append(uniprot_id)
                    else:
                        # 如果不符合，可跳过或警告
                        sys.stderr.write(f"Warning: Possible non-UniProt header: {line}")
                else:
                    # 非标准 FASTA 头部（如 >P12345），尝试提取第一个空格前内容
                    uniprot_id = line[1:].split()[0]
                    uniprot_ids.append(uniprot_id)
    return uniprot_ids

@ck.command()
@ck.option('--fasta-file-name', '-f', default='proteins', help='Input FASTA file')
@ck.option('--data-root', '-dr', default='cafa_data', help='data root directory')
@ck.option('--outfile-name', '-on', default='uniIDs', help='Output file name')
def main(fasta_file_name, data_root, outfile_name):
    for ont in {'mf', 'bp', 'cc'}:
        print(f'Processing {ont} proteins...')
        fasta_file = f'{data_root}/{ont}/{fasta_file_name}.fasta'
        ids = extract_uniprot_ids(fasta_file)
        output_file = f'{data_root}/{ont}/{outfile_name}.txt'
        if output_file:
            with open(output_file, 'w') as out:
                for uid in ids:
                    out.write(uid + '\n')
            print(f"Saved {len(ids)} UniProt IDs to {output_file}")
        else:
            for uid in ids:
                print(uid)

if __name__ == '__main__':
    main()