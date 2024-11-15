import click
import anndata as ad
import logging
import sys
import pickle
import doubletime as dt


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
@click.option('--wgd_depth', type=int, default=0, required=False)
def main(adata_cna, adata_snv, tree_filename, output_cn, output_snv, output_pruned_tree, min_clone_size, min_num_snvs, min_prop_clonal_wgd=0.8, wgd_depth=0):
    adata = ad.read_h5ad(adata_cna)
    adata.obs['haploid_depth'] = adata.obs['coverage_depth'] / adata.obs['ploidy']

    snv_adata = ad.read_h5ad(adata_snv)

    tree = pickle.load(open(tree_filename, 'rb'))

    # make sure the CN adata and SNV adata have the same cells
    adata = adata[snv_adata.obs.index]

    # merge the sbmclone_cluster_id from the SNV adata into the CN adata
    adata.obs = adata.obs.merge(snv_adata.obs[['sbmclone_cluster_id']], left_index=True, right_index=True, how='left')
    assert not adata.obs['sbmclone_cluster_id'].isnull().any()

    # perform automatic WGD depth detection if wgd_depth is negative
    if wgd_depth < 0:
        n_sbmclones = len(adata.obs['sbmclone_cluster_id'].unique())
        if n_sbmclones <= 1:
            # if there is only one clone, set the WGD depth to 0
            # since the root branch is the entire tree
            wgd_depth = 0
        else:
            # otherwise, perform automatic WGD depth detection
            wgd_depth = dt.pp.automatic_wgd_depth_detection(snv_adata)
        print('Automatic WGD depth detection has set wgd_depth to', wgd_depth)

    ### Wrangle CN anndata, identify bins with compatible cn, filter clones

    # Aggregate the CN anndata to go from cell x bin to clone x bin, filtering out clones with too few bins
    # and SNVs that are incompatible with the doubleTime model (i.e. incorrect ASCN state or ASCN not homogenous within a clone)
    adata_cn_clusters, tree, block2leaf = dt.pp.preprocess_cn_adata(adata, tree, min_clone_size=min_clone_size, min_prop_clonal_wgd=min_prop_clonal_wgd)

    # Manually add WGD events to the tree
    # the wgd_depth parameter controls whether a WGD event is added at the root of the tree
    # or multiple independent events are placed on branches `wgd_depth` generations below the root
    dt.pp.add_wgd_tree(tree, adata_cn_clusters, wgd_depth=wgd_depth)

    # split branches with a WGD event into two branches
    dt.pp.split_wgd_branches(tree)

    # Recursively add n_wgd to each clade
    dt.pp.count_wgd(tree.clade, 0)

    ### Wrangle SNV anndata, filter based on cn adatas and merge bin information

    # Make sure that the SNV adata has the same cells as the CN adata
    snv_adata = snv_adata[adata.obs.index]

    # Aggregate the SNV cell x snv anndata to a clone x snv anndata with the same rows and columns
    # as adata_cn_clusters, adding the appropriate layers, obs and var for running the doubleTime model
    adata_clusters = dt.pp.preprocess_snv_adata(snv_adata, adata_cn_clusters, block2leaf)

    adata_cn_clusters.write(output_cn)
    adata_clusters.write(output_snv)
    with open(output_pruned_tree, 'wb') as f:
        pickle.dump(tree, f)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    main()
