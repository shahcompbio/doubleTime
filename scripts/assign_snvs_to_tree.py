import pickle
import click
import pandas as pd
import anndata as ad
import numpy as np
import logging
import sys
import doubletime as dt


def write_empty(output):
    df = pd.DataFrame(columns=[
        'snv', 'leaf', 'alt_counts', 'total_counts', 'cn_state_a', 'cn_state_b', 'clade',
        'vaf', 'snv_id', 'ascn', 'chromosome', 'position', 'ref', 'alt', 'chrom', 'coord', 'tri_nucleotide_context', 
        'is_apobec', 'is_cpg'])
    df.to_csv(output, index=False)


@click.command()
@click.option('--adata')
@click.option('--tree')
@click.option('--output')
@click.option('--ref_genome')
@click.option('--min_total_counts_perblock', type=int, default=2, required=False)
@click.option('--cnloh_only', is_flag=True, default=False, help="Include only SNVs in cnLOH regions")
def main(adata, tree, output, ref_genome, min_total_counts_perblock, cnloh_only):
    if cnloh_only:
        snv_types = ['2:0']
    else:
        snv_types = ['1:0', '1:1', '2:0', '2:1', '2:2']

    # read the input files
    adata = ad.read_h5ad(adata)
    tree = pickle.load(open(tree, 'rb'))

    # Remove non-homogenous snvs and make sure there are still SNVs left
    adata = adata[:, adata.var['is_homogenous_cn']].copy()
    if np.min(adata.shape) == 0:
        print("Anndata has 0 clones or SNVs, shape=", adata.shape)
        write_empty(output)
        return

    # initialize the model object using adata & tree
    model = dt.ml.doubleTimeModel(adata, tree, snv_types=snv_types, min_total_counts_perblock=min_total_counts_perblock)

    # train the model and get the model trace
    model.train()
    model.get_model_trace()

    # get the model output and save it to a csv file
    data = model.format_model_output(ref_genome=ref_genome)
    data.to_csv(output, index = False)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    main()
