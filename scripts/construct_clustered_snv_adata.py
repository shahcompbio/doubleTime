import click
import pandas as pd
import anndata as ad
import numpy as np
import logging
import sys
import pickle
import scgenome


def add_wgd_tree(T, adata_cn_clusters):
    for clade in T.find_clades():
        clade.is_wgd = False

    assert (adata_cn_clusters.obs['n_wgd'] == 0).all() or (adata_cn_clusters.obs['n_wgd'] == 1).all()
    T.clade.is_wgd = (adata_cn_clusters.obs['n_wgd'] == 1).all()


def count_wgd(clade, n_wgd):
    if clade.is_wgd:
        clade.n_wgd = n_wgd + 1
    else:
        clade.n_wgd = n_wgd
    for child in clade.clades:
        count_wgd(child, clade.n_wgd)


@click.command()
@click.option('--adata_cna')
@click.option('--adata_snv')
@click.option('--tree_filename')
@click.option('--output_cn')
@click.option('--output_snv')
@click.option('--output_pruned_tree')
@click.option('--min_clone_size', type=int, default=20, required=False)
@click.option('--min_num_snvs', type=int, default=20, required=False) # TODO
@click.option('--min_prop_clonal_wgd', type=float, default=0.8, required=False)
def main(adata_cna, adata_snv, tree_filename, output_cn, output_snv, output_pruned_tree, min_clone_size, min_num_snvs, min_prop_clonal_wgd):
    adata = ad.read_h5ad(adata_cna)
    adata.obs['haploid_depth'] = adata.obs['coverage_depth'] / adata.obs['ploidy']

    snv_adata = ad.read_h5ad(adata_snv)

    tree = pickle.load(open(tree_filename, 'rb'))

    adata = adata[snv_adata.obs.index]
    adata.obs = adata.obs.merge(snv_adata.obs[['sbmclone_cluster_id']], left_index=True, right_index=True, how='left')
    assert not adata.obs['sbmclone_cluster_id'].isnull().any()

    # Wrangle CN anndata, identify bins with compatible cn, filter clones
    # 

    # Add the modal wgd state for each sbmclone
    adata.obs['n_wgd_mode'] = adata.obs.groupby('sbmclone_cluster_id')['n_wgd'].transform(lambda x: x.mode()[0])

    # Add leaf id to copy number anndata
    # Since multiple clones could have been combined into one leaf
    # of the clone tree, there is not necessarily a one to one mapping
    # of leaves to clones
    block2leaf = {}
    for l in tree.get_terminals():
        for b in l.name.lstrip('clone_').split('/'):
            block2leaf[int(b)] = l.name.lstrip('clone_') # TODO: why?
    adata.obs['leaf_id'] = adata.obs.sbmclone_cluster_id.map(block2leaf)

    # Select cells with n_wgd<=1 and n_wgd equal to the modal n_wgd
    adata = adata[(
        (adata.obs.n_wgd.astype(int) <= 1) &
        adata.obs.n_wgd == adata.obs.n_wgd_mode)].copy()

    # Threshold on size of clone
    adata.obs['leaf_size'] = adata.obs.groupby('leaf_id').transform('size')
    adata = adata[adata.obs['leaf_size'] >= min_clone_size]

    # Aggregate the copy number based on the leaf id
    adata_cn_clusters = scgenome.tl.aggregate_clusters(
        adata,
        agg_layers={
            'Maj': 'median',
            'Min': 'median',
            'state': 'median',
        },
        cluster_col='leaf_id')
    adata_cn_clusters.obs.index = adata_cn_clusters.obs.index.astype(str)

    # Add per clone statistics for the frequency of the median state
    for layer in ['state', 'Maj', 'Min']:
        adata.layers[f'clone_median_{layer}'] = adata_cn_clusters.to_df(layer).loc[adata.obs['leaf_id'].values, :]
        adata.layers[f'is_eq_clone_median_{layer}'] = adata.layers[layer] == adata.layers[f'clone_median_{layer}']
        adata.var[f'is_eq_clone_median_{layer}'] = np.nanmean(adata.layers[f'is_eq_clone_median_{layer}'], axis=0)

    # Redo aggregation of the copy number and include per clone stats for frequency of median state
    adata_cn_clusters = scgenome.tl.aggregate_clusters(
        adata,
        agg_layers={
            'Maj': 'median',
            'Min': 'median',
            'state': 'median',
            'is_eq_clone_median_state': 'mean',
            'is_eq_clone_median_Maj': 'mean',
            'is_eq_clone_median_Min': 'mean',
        },
        agg_obs={
            'n_wgd': 'median',
            'haploid_depth': 'sum',
        },
        cluster_col='leaf_id')
    adata_cn_clusters.obs.index = adata_cn_clusters.obs.index.astype(str)
    adata_cn_clusters.obs['n_wgd'] = adata_cn_clusters.obs['n_wgd'].round()

    # Calculate and threshold for homogeneous copy number within each clone
    adata_cn_clusters.var['is_homogenous_cn'] = (
        (adata_cn_clusters.var['is_eq_clone_median_Maj'] > 0.9) &
        (adata_cn_clusters.var['is_eq_clone_median_Min'] > 0.9) &
        (adata_cn_clusters.layers['is_eq_clone_median_Maj'] > 0.8).all(axis=0) &
        (adata_cn_clusters.layers['is_eq_clone_median_Min'] > 0.8).all(axis=0))

    # Compatible states for WGD1 and WGD0, major/minor for the snv tree model
    compatible_cn_types = {
        '1:0': [{'n_wgd': 1, 'Maj': 1, 'Min': 0}, {'n_wgd': 0, 'Maj': 1, 'Min': 0}],
        '2:0': [{'n_wgd': 1, 'Maj': 2, 'Min': 0}, {'n_wgd': 0, 'Maj': 1, 'Min': 0}],
        '1:1': [{'n_wgd': 1, 'Maj': 1, 'Min': 1}, {'n_wgd': 0, 'Maj': 1, 'Min': 1}],
        '2:1': [{'n_wgd': 1, 'Maj': 2, 'Min': 1}, {'n_wgd': 0, 'Maj': 1, 'Min': 1}],
        '2:2': [{'n_wgd': 1, 'Maj': 2, 'Min': 2}, {'n_wgd': 0, 'Maj': 1, 'Min': 1}],
    }

    # Check for compatibility and assign
    adata_cn_clusters.var['snv_type'] = 'incompatible'
    for name, cn_states in compatible_cn_types.items():
        cn_states = pd.DataFrame(cn_states).set_index('n_wgd')
        clone_maj = adata_cn_clusters.obs['n_wgd'].map(cn_states['Maj'])
        clone_min = adata_cn_clusters.obs['n_wgd'].map(cn_states['Min'])
        is_compatible = (
            (adata_cn_clusters.layers['Maj'] == clone_maj.values[:, np.newaxis]) &
            (adata_cn_clusters.layers['Min'] == clone_min.values[:, np.newaxis]))
        bin_is_compatible = np.all(is_compatible, axis=0)
        adata_cn_clusters.var.loc[bin_is_compatible, 'snv_type'] = name

    # Wrangle tree, prune based on removed clusters
    #

    # Prune clones from the tree if they were removed due to size
    remaining_leaves = adata_cn_clusters.obs.index
    tree = scgenome.tl.prune_leaves(tree, lambda a: a.name.lstrip('clone_') not in remaining_leaves)
    # Merge branches
    def merge_branches(parent, child):
        return {
            'name': child.name,
            'branch_length': 1,
            'mutations': ','.join(filter(lambda a: a, [parent.mutations, child.mutations])),
        }
    tree = scgenome.tl.aggregate_tree_branches(tree, f_merge=merge_branches)

    # Manually add WGD events to the tree
    add_wgd_tree(tree, adata_cn_clusters)

    # Recursively add n_wgd to each clade
    count_wgd(tree.clade, 0)

    # Wrangle SNV anndata, filter based on cn adatas and merge bin information
    #

    # Filter snv adata similarly to cn adata
    snv_adata = snv_adata[adata.obs.index]

    # Aggregate snv counts
    snv_adata.obs['leaf_id'] = snv_adata.obs.sbmclone_cluster_id.map(block2leaf)
    adata_clusters = scgenome.tl.aggregate_clusters(
        snv_adata,
        agg_layers={
            'alt': 'sum',
            'ref': 'sum',
            'Maj': 'median',
            'Min': 'median',
            'state': 'median',
        },
        cluster_col='leaf_id')

    # Filter snv cluster data similar to copy number
    adata_clusters = adata_clusters[adata_cn_clusters.obs.index].copy()

    # Additional layers
    adata_clusters.layers['vaf'] = adata_clusters.layers['alt'] / (adata_clusters.layers['ref'] + adata_clusters.layers['alt'])
    adata_clusters.layers['ref_count'] = adata_clusters.layers['ref']
    adata_clusters.layers['alt_count'] = adata_clusters.layers['alt']
    adata_clusters.layers['total_count'] = adata_clusters.layers['ref'] + adata_clusters.layers['alt']

    # Add information from cn bin analysis
    adata_clusters.var['snv_type'] = adata_cn_clusters.var.loc[adata_clusters.var['cn_bin'], 'snv_type'].values
    adata_clusters.var['is_homogenous_cn'] = adata_cn_clusters.var.loc[adata_clusters.var['cn_bin'], 'is_homogenous_cn'].values

    adata_cn_clusters.write(output_cn)
    adata_clusters.write(output_snv)
    with open(output_pruned_tree, 'wb') as f:
        pickle.dump(tree, f)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    main()
