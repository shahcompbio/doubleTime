import os
import itertools
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import anndata as ad
import Bio.Phylo
import numpy as np
from copy import deepcopy
import wgs_analysis.snvs.mutsig
import sys
import click
import logging


def plot_snv_reads_hist(adata, snv_reads_hist_filename, min_total_counts_perblock=2):
    adata.var['min_total_count'] = np.array(adata.layers['total_count'].min(axis=0))

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    adata.var['min_total_count'].hist(bins=20, ax=ax)

    # compute the nubmer of SNVs that pass the filter in all clones
    n = (adata.var['min_total_count'] >= min_total_counts_perblock).sum()

    ax.set_xlabel("Minimum total reads")
    ax.set_ylabel("Count")
    ax.set_title(f"SNVs that pass filter\nn={n} SNVs with at least {min_total_counts_perblock} reads in all clones")
    fig.savefig(snv_reads_hist_filename, bbox_inches='tight')

    # filter the SNVs according to the minimum total count
    adata = adata[:, adata.var['min_total_count'] >= min_total_counts_perblock]
    return adata


@click.command()
@click.option('--tree_filename')
@click.option('--adata_filename')
@click.option('--table_filename')
@click.option('--patient_id')
@click.option('--snv_reads_hist_filename', '-srh')
def main(tree_filename, adata_filename, table_filename, patient_id, snv_reads_hist_filename):
    # load the input files
    adata = ad.read_h5ad(adata_filename)
    tree = pickle.load(open(tree_filename, 'rb'))
    data = pd.read_csv(table_filename)

    # check that the adata object is not empty
    if np.min(adata.shape) == 0 or len(data) == 0:
        sys.exit(0)
    
    # plot the SNV reads histogram
    adata = plot_snv_reads_hist(adata, snv_reads_hist_filename)
    


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    main()