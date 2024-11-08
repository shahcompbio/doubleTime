import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import anndata as ad
import Bio.Phylo
import numpy as np
from copy import deepcopy
import sys
import click
import logging
import doubletime as dt


@click.command()
@click.option('--tree_filename')
@click.option('--adata_filename')
@click.option('--table_filename')
@click.option('--patient_id')
@click.option('--snv_reads_hist_filename', '-srh')
@click.option('--clone_hist_filename', '-ch')
@click.option('--clone_pairwise_vaf_filename', '-cpv')
@click.option('--snv_multiplicity_filename', '-sm')
@click.option('--bio_phylo_tree_filename', '-bpt')
@click.option('--wgd_tree_filename', '-wt')
@click.option('--bio_phylo_cpg_tree_filename', '-bptc')
@click.option('--cpg_tree_filename', '-ct')
@click.option('--apobec_tree_filename', '-at')
def main(tree_filename, adata_filename, table_filename, patient_id, 
         snv_reads_hist_filename, clone_hist_filename, clone_pairwise_vaf_filename,
         snv_multiplicity_filename, bio_phylo_tree_filename, wgd_tree_filename, 
         bio_phylo_cpg_tree_filename, cpg_tree_filename, apobec_tree_filename):
    # load the input files
    adata = ad.read_h5ad(adata_filename)
    tree = pickle.load(open(tree_filename, 'rb'))
    data = pd.read_csv(table_filename)

    # plot the clone histogram
    g = dt.pl.plot_clone_hist(data)
    g.savefig(clone_hist_filename, bbox_inches='tight')

    # plot the pairwise VAF
    g = dt.pl.plot_clone_pairwise_vaf(data)
    g.savefig(clone_pairwise_vaf_filename, bbox_inches='tight')

    # plot the SNV multiplicity
    # VAF for each SNV multiplicity
    g = dt.pl.plot_snv_hist_facetgrid(data, col='cn_state_a', row=None, x='vaf', hue='ascn')
    g.savefig(snv_multiplicity_filename, bbox_inches='tight')

    # VAF for each clade (row) and each clone (column)
    g = dt.pl.plot_snv_hist_facetgrid(data, col='leaf', row='clade', x='vaf', hue='ascn')
    output_filename = snv_multiplicity_filename.replace('.pdf', '_by_clone.pdf')
    g.savefig(output_filename, bbox_inches='tight')

    # Plot VAFs for the zero state
    zero_state_data = data.query('cn_state_a == 0 & cn_state_b == 0')
    # VAF for the zero state
    if len(zero_state_data) > 0:
        g = dt.pl.plot_snv_hist_facetgrid(zero_state_data, col='leaf', row=None, x='vaf', hue='ascn')
        output_filename = snv_multiplicity_filename.replace('.pdf', '_zero_state.pdf')
        g.savefig(output_filename, bbox_inches='tight')
    
    # Pairwise VAF for pairs of clones for the zero state only
    if len(zero_state_data) > 0:
        plot_data = zero_state_data.set_index(['snv', 'clade', 'leaf'])['vaf'].unstack().reset_index(level=1)
        plot_data['clade'] = plot_data['clade'].astype('category')
        g = sns.pairplot(data=plot_data, hue='clade')
        output_filename = snv_multiplicity_filename.replace('.pdf', '_zero_state_by_clone.pdf')
        g.savefig(output_filename, bbox_inches='tight')

    # check that the adata object is not empty
    if np.min(adata.shape) == 0 or len(data) == 0:
        raise Exception('Empty adata object')

    # plot the SNV reads histogram
    fig, ax = dt.pl.plot_snv_reads_hist(adata)
    fig.savefig(snv_reads_hist_filename, bbox_inches='tight')

    # filter the SNVs according to the minimum total count
    adata = adata[:, adata.var['min_total_count'] >= 2]

    # filter adata based on tree
    clones = []
    for leaf in tree.get_terminals():
        clones.append(leaf.name)
    adata = adata[clones].copy()

    # find the SNV types included in the tree
    snv_types = sorted(data.ascn.unique())

    ## Plot the tree with the branch lengths annotated by the number of SNVs
    ## and WGD events represented by color
    # compute the number of cells assigned to each clone
    cell_counts = adata.obs['cluster_size'].copy()
    # compute the branch lengths of each clade
    branch_lengths = dt.tl.compute_branch_lengths(data, tree, cell_counts)

    # draw a doubleTime tree using all SNV counts as branch lengths
    ax = dt.pl.plot_clone_tree(tree, branch_lengths, cell_counts)
    ax.set_title(f'{patient_id}\nSNV types: {snv_types}', fontsize=10)
    fig = ax.get_figure()
    fig.savefig(wgd_tree_filename, bbox_inches='tight')

    # draw a Bio.Phylo tree using the SNV counts as branch lengths
    Bio.Phylo.draw(tree)
    plt.savefig(bio_phylo_tree_filename, bbox_inches='tight')

    ## Plot the tree with the branch lengths annotated by the number of CpG SNVs
    cpg_tree = deepcopy(tree)
    # compute the branch lengths of each clade based on CpG SNVs
    branch_lengths_cpg = dt.tl.compute_branch_lengths(data, cpg_tree, cell_counts, CpG=True)
    # draw a doubleTime tree using CpG SNV counts as branch lengths
    ax = dt.pl.plot_clone_tree(cpg_tree, branch_lengths_cpg, cell_counts)
    ax.set_xlabel('# CpG SNVs')
    ax.set_title(f'{patient_id}\nSNV types: {snv_types}', fontsize=10)
    fig = ax.get_figure()
    fig.savefig(cpg_tree_filename, bbox_inches='tight')

    # draw a Bio.Phylo tree using the CpG SNV counts as branch lengths
    Bio.Phylo.draw(cpg_tree)
    plt.savefig(bio_phylo_cpg_tree_filename, bbox_inches='tight')

    # compute the fraction of ABOPEC SNVs in each clade
    apobec_fraction = dt.tl.compute_clade_apobec_fraction(data)

    # plot the tree with branch lengths annotated by APOBEC fraction
    ax = dt.pl.plot_clone_tree(tree, branch_lengths, cell_counts, apobec_fraction=apobec_fraction)
    ax.set_title(f'{patient_id} APOBEC\nSNV types: {snv_types}', fontsize=10)
    fig = ax.get_figure()
    fig.savefig(apobec_tree_filename, bbox_inches='tight')


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    main()
