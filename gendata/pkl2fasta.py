import os
import sys
import click as ck
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

@ck.command()
@ck.option('--data-root', '-dr', default='data', required=True, help='data root')
@ck.option('--file-name', '-fn', default='proteins', required=True, help='pickle file name')
def main(data_root, file_name):
    for ont in ['mf']:
        print(f'Processing {ont} proteins...')
        pkl_file = f'{data_root}/{ont}/{file_name}.pkl'
        fasta_file = f'{data_root}/{ont}/{file_name}.fasta'
        df = pd.read_pickle(pkl_file)
        if 'proteins' not in df.columns or 'sequences' not in df.columns:
            raise ValueError("DataFrame must contain 'proteins' and 'sequences' columns.")

        with open(fasta_file, 'w') as f:
            for _, row in df.iterrows():
                protein_id = row['proteins']
                sequence = row['sequences']
                f.write(f">{protein_id}\n{sequence}\n")

        print(f"Successfully wrote {len(df)} sequences to {fasta_file}")


if __name__ == '__main__':
    main()
