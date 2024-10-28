import itertools
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
import itertools
import matplotlib.patches as patches
import doubletime as dt

n_wgd_colors = dt.pl.n_wgd_colors


def plot_snv_multiplicity(data, output_filename1):
    # VAF for each SNV multiplicity
    g = sns.FacetGrid(col='cn_state_a', data=data, sharey=False, hue='ascn')
    g.map_dataframe(sns.histplot, x='vaf', bins=20, binrange=(0, 1))
    g.add_legend()
    g.savefig(output_filename1, bbox_inches='tight')

    # VAF for each clade (row) and each clone (column)
    g = sns.FacetGrid(col='leaf', row='clade', data=data, sharey=False, hue='ascn')
    g.map_dataframe(sns.histplot, x='vaf', bins=20, binrange=(0, 1))
    g.add_legend()
    output_filename2 = output_filename1.replace('.pdf', '_by_clone.pdf')
    g.savefig(output_filename2, bbox_inches='tight')

    # Plot VAFs for the zero state
    zero_state_data = data.query('cn_state_a == 0 & cn_state_b == 0')
    # VAF for the zero state
    if len(zero_state_data) > 0:
        g = sns.FacetGrid(col='leaf', data=zero_state_data, sharey=False, hue='ascn')
        g.map_dataframe(sns.histplot, x='vaf', bins=20, binrange=(0, 1))
        g.add_legend()
        output_filename3 = output_filename1.replace('.pdf', '_zero_state.pdf')
        g.savefig(output_filename3, bbox_inches='tight')
    
    # Pairwise VAF for pairs of clones for the zero state only
    if len(zero_state_data) > 0:
        plot_data = zero_state_data.set_index(['snv', 'clade', 'leaf'])['vaf'].unstack().reset_index(level=1)
        plot_data['clade'] = plot_data['clade'].astype('category')
        g = sns.pairplot(data=plot_data, hue='clade')
        output_filename4 = output_filename1.replace('.pdf', '_zero_state_by_clone.pdf')
        g.savefig(output_filename4, bbox_inches='tight')


# def draw_branch_wgd(ax, clade, bar_height=0.25):
#     if clade.is_wgd:
#         length1 = clade.branch_length * clade.wgd_fraction
#         length2 = clade.branch_length * (1. - clade.wgd_fraction)
#         rect1 = patches.Rectangle((clade.branch_start, clade.branch_pos-bar_height/2.), length1, bar_height, 
#                                   linewidth=0, edgecolor='none', facecolor=n_wgd_colors[clade.n_wgd-1])
#         ax.add_patch(rect1)
#         rect2 = patches.Rectangle((clade.branch_start + length1, clade.branch_pos-bar_height/2.), length2, bar_height, 
#                                   linewidth=0, edgecolor='none', facecolor=n_wgd_colors[clade.n_wgd])
#         ax.add_patch(rect2)
#         ax.scatter([clade.branch_start + length1], [clade.branch_pos+bar_height], marker='v', color='darkorange')
#     else:
#         rect = patches.Rectangle(
#             (clade.branch_start, clade.branch_pos-bar_height/2.), clade.branch_length, bar_height, 
#             linewidth=0, edgecolor='none', facecolor=n_wgd_colors[clade.n_wgd])
#         ax.add_patch(rect)


# def draw_branch_links(ax, clade, bar_height=0.25):
#     if not clade.is_terminal():
#         child_pos = [child.branch_pos for child in clade.clades]
#         plt.plot(
#             [clade.branch_start + clade.branch_length, clade.branch_start + clade.branch_length],
#             [min(child_pos)-bar_height/2., max(child_pos)+bar_height/2.], color='k', ls=':')


# def draw_leaf_tri_size(ax, clade, bar_height=0.25, max_height=2.):
#     if clade.is_terminal():
#         expansion_height = bar_height + max(0.1, clade.cell_fraction) * (max_height - bar_height) # bar_height to 1.5

#         # Transform to create a regular shaped triangle
#         height = (ax.transData.transform([0, expansion_height]) - ax.transData.transform([0, 0]))[1]
#         length = (ax.transData.inverted().transform([height, 0]) - ax.transData.inverted().transform([0, 0]))[0]

#         branch_end = clade.branch_start+clade.branch_length
#         branch_pos_bottom = clade.branch_pos-bar_height/2.
#         branch_pos_top = clade.branch_pos+bar_height/2.

#         vertices = [
#             [branch_end, branch_pos_bottom],
#             [branch_end, branch_pos_top],
#             [branch_end + length, branch_pos_top + expansion_height / 2],
#             [branch_end + length, branch_pos_bottom - expansion_height / 2],
#         ]
#         tri = patches.Polygon(vertices, linewidth=1, edgecolor='0.25', facecolor='0.25')
#         ax.add_patch(tri)


# def draw_branch_wgd_event(ax, clade, bar_height=0.25):
#     if clade.is_wgd:
#         length1 = clade.branch_length * clade.wgd_fraction
#         ax.scatter([clade.branch_start + length1], [clade.branch_pos+bar_height], marker='v', color='darkorange')






# def plot_cpg_tree(tree, cpg_tree, patient_id, snv_types, output_filename):
#     fig, ax = plt.subplots(figsize=(5, 1), dpi=150)

#     for clade in cpg_tree.find_clades():
#         draw_branch_wgd(ax, clade)
#         draw_branch_links(ax, clade)

#     yticks = list(zip(*[(clade.branch_pos, f'{clade.name.replace("_", " ")}, n={clade.cell_count}') for clade in tree.get_terminals()]))
#     ax.set_yticks(*yticks)
#     ax.yaxis.tick_right()
#     sns.despine(trim=True, left=True, right=False)
#     ax.yaxis.tick_right()
#     ax.yaxis.set_ticks_position('right')
#     ax.set_xlabel('# CpG SNVs')
#     ax.set_title(f'{patient_id}\n SNV types: {snv_types}', fontsize=10)


#     for clade in cpg_tree.find_clades():
#         draw_leaf_tri_size(ax, clade, bar_height=0)


#     legend_elements = [patches.Patch(color=n_wgd_colors[0], label='0'),
#                     patches.Patch(color=n_wgd_colors[1], label='1'),
#                     patches.Patch(color=n_wgd_colors[2], label='2')]
#     legend_1 = ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.5, 1), frameon=False, fontsize=8, title='#WGD')
#     fig.savefig(output_filename, bbox_inches='tight')


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
# def main(tree_filename, adata_filename, table_filename, patient_id, 
#          snv_reads_hist_filename, clone_hist_filename, clone_pairwise_vaf_filename,
#          snv_multiplicity_filename, bio_phylo_tree_filename, wgd_tree_filename,
#          apobec_tree_filename, bio_phylo_cpg_tree_filename, cpg_tree_filename):
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
    # TODO: move this function to doubletime.plotting module
    plot_snv_multiplicity(data, snv_multiplicity_filename)

    # check that the adata object is not empty
    if np.min(adata.shape) == 0 or len(data) == 0:
        sys.exit(0)

    # plot the SNV reads histogram
    fig, ax = dt.pl.plot_snv_reads_hist(adata)
    fig.savefig(snv_reads_hist_filename, bbox_inches='tight')

    # filter the SNVs according to the minimum total count
    adata = adata[:, adata.var['min_total_count'] >= 2]

    # filter adata based on tree
    clones = []
    for leaf in tree.get_terminals():
        clones.append(leaf.name.replace('clone_', '').replace('postwgd_', ''))
    adata = adata[clones].copy()

    # find the SNV types included in the tree
    snv_types = sorted(data.ascn.unique())

    ## Plot the tree with the branch lengths annotated by the number of SNVs
    ## and WGD events represented by color
    # compute the number of cells assigned to each clone
    cell_counts = dt.pl.compute_cell_counts(adata, tree)
    # compute the branch lengths of each clade
    branch_lengths = dt.pl.compute_branch_lengths(data, tree, cell_counts)

    # draw a doubleTime tree using all SNV counts as branch lengths
    ax = dt.pl.plot_clone_tree(tree, branch_lengths, cell_counts)
    ax.set_xlabel('# SNVs')
    ax.set_title(f'{patient_id}\nSNV types: {snv_types}', fontsize=10)
    fig = ax.get_figure()
    fig.savefig(wgd_tree_filename, bbox_inches='tight')

    # draw a Bio.Phylo tree using the SNV counts as branch lengths
    Bio.Phylo.draw(tree)
    plt.savefig(bio_phylo_tree_filename, bbox_inches='tight')

    ## Plot the tree with the branch lengths annotated by the number of CpG SNVs
    cpg_tree = deepcopy(tree)
    # compute the branch lengths of each clade based on CpG SNVs
    branch_lengths_cpg = dt.pl.compute_branch_lengths(data, cpg_tree, cell_counts, CpG=True)
    # draw a doubleTime tree using CpG SNV counts as branch lengths
    ax = dt.pl.plot_clone_tree(cpg_tree, branch_lengths_cpg, cell_counts)
    ax.set_xlabel('# CpG SNVs')
    ax.set_title(f'{patient_id}\nSNV types: {snv_types}', fontsize=10)
    fig = ax.get_figure()
    fig.savefig(cpg_tree_filename, bbox_inches='tight')

    # draw a Bio.Phylo tree using the CpG SNV counts as branch lengths
    Bio.Phylo.draw(cpg_tree)
    plt.savefig(bio_phylo_cpg_tree_filename, bbox_inches='tight')

    # TODO: fix apobec plotting and add it to doubletime.plotting module
    apobec_fraction = data[['snv', 'clade', 'is_apobec']].drop_duplicates().groupby(['clade'])['is_apobec'].mean()
    print('apobec_fraction:', apobec_fraction, sep='\n')

    # plot the tree with branch lengths annotated by APOBEC fraction
    ax = dt.pl.plot_apobec_tree(tree, branch_lengths, cell_counts, apobec_fraction)
    ax.set_title(f'{patient_id} APOBEC', fontsize=10)
    fig = ax.get_figure()
    fig.savefig(apobec_tree_filename, bbox_inches='tight')

    # # Compute branch lengths as SNV counts (preferred)
    # clone_sizes = adata.obs['cluster_size'].copy()
    # clone_sizes.index = [f'clone_{a}' for a in clone_sizes.index]
    # for clade in tree.find_clades():
    #     clade_df = data[data.clade == clade.name]
    #     cntr = clade_df.wgd_timing.value_counts()
    #     clade.branch_length = len(clade_df.snv_id.unique())
    #     if 'prewgd' in cntr.index:
    #         assert 'postwgd' in cntr.index
    #         clade.wgd_fraction = 2 * cntr['prewgd'] / (2 * cntr['prewgd'] + cntr['postwgd'])
    #     if clade.is_terminal():
    #         clade.cell_count = clone_sizes.loc[clade.name]
    #         clade.cell_fraction = clone_sizes.loc[clade.name] / clone_sizes.sum()

    

    # # count the number of WGD events in each clade
    # dt.tl.count_wgd(tree.clade, 0)


    # # plot the tree with total SNV counts as branch lengths
    # # WGD timing on branches is annotated by color
    # snv_types = sorted(data.ascn.unique())
    # dt.pl.assign_plot_locations(tree)
    # plot_wgd_tree(tree, patient_id, snv_types, wgd_tree_filename)
    
    # # compute the fraction of ABOPEC SNVs in each clade
    # apobec_fraction = data[['snv', 'clade', 'wgd_timing', 'is_apobec']].drop_duplicates().groupby(['clade', 'wgd_timing'])['is_apobec'].mean()
    
    # # plot the tree with branch lengths annotated by APOBEC fraction
    # plot_apobec_tree(tree, apobec_fraction, patient_id, apobec_tree_filename)

    # # restrict to CpG SNVs
    # cpg_tree = deepcopy(tree)
    # for clade in cpg_tree.find_clades():
    #     clade_df = data[(data.clade == clade.name) & (data.is_cpg)]
        
    #     clade.branch_length = len(clade_df.snv_id.unique())
        
    #     if clade.is_wgd:
    #         cntr = clade_df.wgd_timing.value_counts().reindex(['prewgd', 'postwgd'])
    #         clade.wgd_fraction = 2 * cntr['prewgd'] / (2 * cntr['prewgd'] + cntr['postwgd'])
    #     if clade.is_terminal():
    #         clade.cell_count = clone_sizes.loc[clade.name]
    #         clade.cell_fraction = clone_sizes.loc[clade.name] / clone_sizes.sum()

    # # draw a Bio.Phylo tree using the CpG SNV counts as branch lengths
    # Bio.Phylo.draw(cpg_tree)
    # plt.savefig(bio_phylo_cpg_tree_filename, bbox_inches='tight')

    # # plot the tree using CpG SNVs
    # # WGD timing on branches are annotated by color
    # assign_plot_locations(cpg_tree)
    # plot_cpg_tree(tree, cpg_tree, patient_id, snv_types, cpg_tree_filename)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    main()
