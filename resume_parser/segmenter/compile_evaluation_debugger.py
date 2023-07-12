import argparse
import glob
import os
import pandas as pd
import numpy as np

from argparse import Namespace
from typing import List

def combine_dfs(evaluation_dfs: List[pd.DataFrame], float_format: str) -> pd.DataFrame:
    df_concat = pd.concat(evaluation_dfs)
    group_df = df_concat.groupby(level=0)

    df_means = group_df.mean().applymap(float_format.format)
    df_stds = group_df.std().applymap(float_format.format)
    df = df_means.combine(df_stds, lambda x1, x2: x1 + ' ± ' + x2)
    return df


def combine_item_segment_dfs(evaluation_dfs: List[pd.DataFrame], float_format: str) -> pd.DataFrame:
    # Only keep the Work_Experience and Education items
    evaluation_dfs = [df[df.index.isin(['Work_Experience', 'Education'])]
                      for df in evaluation_dfs]
    for df in evaluation_dfs:
        df.loc['macro'] = df.mean(axis=0)
    return combine_dfs(evaluation_dfs, float_format)


def get_sheets(filenames: List[str],
               sheet_name: str,
               index_col: List[int]
              ) -> List[pd.DataFrame]:
    dfs = []
    for filename in filenames:
        df = pd.read_excel(filename, sheet_name=sheet_name)
        columns_names = df.columns[index_col].tolist()
        if 'filename' in columns_names:
            columns_names.remove('filename')
            df = df[df['filename'] == 'ALL']
            df.drop(columns=['filename'], inplace=True)
        if len(columns_names) > 0:
            df.set_index(columns_names, inplace=True)
        dfs.append(df)
    return dfs


def main(args: Namespace):
    pattern = os.path.join(args.experiment_dir, fr'split_*/{args.filename_pattern}')
    filenames = glob.glob(pattern)

    entries = [
        ('classification', [0,1], '{:.4f}'),
        ('classification (no iob)', [0,1], '{:.4f}'),
        ('classification (segment)', [0,1], '{:.4f}'),
        ('classification (section segment)', [0,1], '{:.4f}'),
        ('confusion matrix', [0], '{:.0f}'),
        ('confusion matrix (no iob)', [0], '{:.0f}'),
        ('segmentation', [0], '{:.4f}'),
    ]

    output_filename = os.path.join(args.experiment_dir, 'evaluation.xlsx')
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        for sheet_name, index_col, float_format in entries:
            evaluation_dfs = get_sheets(filenames, sheet_name, index_col)
            combined_df = combine_dfs(evaluation_dfs, float_format)
            combined_df.to_excel(writer, sheet_name=sheet_name)

        # Item segmentation
        sheet_name, index_col, float_format = ('classification (segment)', [0,1], '{:.4f}')
        evaluation_dfs = get_sheets(filenames, sheet_name, index_col)
        combined_df = combine_item_segment_dfs(evaluation_dfs, float_format)
        combined_df.to_excel(writer, sheet_name='classification (item segment)')


def get_args() -> Namespace:
    parser = argparse.ArgumentParser()

    # Testing related
    parser.add_argument('--experiment_dir', type=str, default=None,
                        help='Experiment directory. If provided, test mode is enabled '
                        'and the model will be loaded from this directory.')
    parser.add_argument('--filename_pattern', type=str, default='*_evaluation.xlsx',
                        help='Filename pattern for evaluation files.')
    return parser.parse_args()

if __name__ == '__main__':
    main(get_args())
