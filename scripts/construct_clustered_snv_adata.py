import click
import pandas as pd
import anndata as ad
import numpy as np
import logging
import sys
import pickle
import scipy
import math
import itertools
import warnings
from scipy.stats import binom, gaussian_kde, poisson
import scgenome
import doubletime as dt


def get_blocks_adata(adata, cluster_col, epsilon = 0.001):
    blocks_adata = scgenome.tl.aggregate_clusters(
        adata[~adata.obs[cluster_col].isna()],
        agg_layers={
            'alt': np.sum,
            'ref': np.sum,
            'Maj': np.median,
            'Min': np.median,
            'state': np.median,
            'total': np.sum
        },
        cluster_col=cluster_col)
    blocks_adata.layers['vaf'] = blocks_adata.layers['alt'] / (blocks_adata.layers['total'])
    blocks_adata.layers['p_cn0'] = binom.logpmf(k=blocks_adata.layers['alt'], n=blocks_adata.layers['total'], p = epsilon)
    blocks_adata.layers['p_cn1'] = binom.logpmf(k=blocks_adata.layers['alt'], n=blocks_adata.layers['total'], p = 0.5)
    blocks_adata.layers['p_cn2'] = binom.logpmf(k=blocks_adata.layers['alt'], n=blocks_adata.layers['total'], p = 1-epsilon)

    # subset blocks_adata to only the cnloh SNVs (those with Maj=2 and Min=0 across all clones)
    # only CNLOH SNVs should be used to infer the WGD depth
    blocks_adata.var['is_cnloh'] = np.all((blocks_adata.layers['Maj'] == 2) & (blocks_adata.layers['Min'] == 0), axis=0)
    blocks_adata = blocks_adata[:, blocks_adata.var['is_cnloh']]

    return blocks_adata

def get_partition(blocks_adata, partition, epsilon = 0.001, min_snvcov_reads = 2):
    assert len(partition) == len(blocks_adata), (len(partition), len(blocks_adata))
    partition = np.array(partition)
    part1_idx = np.where(partition == 1)[0]
    part2_idx = np.where(partition == 2)[0]
    assert len(part1_idx) > 0 and len(part2_idx) > 0, (part1_idx, part2_idx)
    
    new_blocks_adata = blocks_adata[:2].copy()
    new_blocks_adata.layers['alt'][0] = blocks_adata[part1_idx].layers['alt'].toarray().sum(axis = 0)
    new_blocks_adata.layers['ref'][0] = blocks_adata[part1_idx].layers['ref'].toarray().sum(axis = 0)
    new_blocks_adata.layers['total'][0] = blocks_adata[part1_idx].layers['total'].toarray().sum(axis = 0)
    new_blocks_adata.layers['Min'][0] = np.median(blocks_adata[part1_idx].layers['Min'], axis = 0)
    new_blocks_adata.layers['Maj'][0] = np.median(blocks_adata[part1_idx].layers['Maj'], axis = 0)
    new_blocks_adata.layers['state'][0] = np.median(blocks_adata[part1_idx].layers['state'], axis = 0)
    new_blocks_adata.obs['partition_size'] = [sum([blocks_adata.obs.cluster_size[i] for i in part1_idx]),
                                              sum([blocks_adata.obs.cluster_size[i] for i in part2_idx])]
    new_blocks_adata.obs['blocks'] = ['/'.join([blocks_adata.obs.iloc[a].name for a in l]) for l in [part1_idx, part2_idx]]
    
    new_blocks_adata.layers['alt'][1] = blocks_adata[part2_idx].layers['alt'].toarray().sum(axis = 0)
    new_blocks_adata.layers['ref'][1] = blocks_adata[part2_idx].layers['ref'].toarray().sum(axis = 0)
    new_blocks_adata.layers['total'][1] = blocks_adata[part2_idx].layers['total'].toarray().sum(axis = 0)
    new_blocks_adata.layers['Min'][1] = np.median(blocks_adata[part2_idx].layers['Min'], axis = 0)
    new_blocks_adata.layers['Maj'][1] = np.median(blocks_adata[part2_idx].layers['Maj'], axis = 0)
    new_blocks_adata.layers['state'][1] = np.median(blocks_adata[part2_idx].layers['state'], axis = 0)
    
    new_blocks_adata.layers['vaf'] = new_blocks_adata.layers['alt'] / (new_blocks_adata.layers['total'])
    
    # add layers for marginal probabilities of CN states
    new_blocks_adata.layers['p_cn0'] = binom.logpmf(k=new_blocks_adata.layers['alt'], n=new_blocks_adata.layers['total'], p = epsilon)
    new_blocks_adata.layers['p_cn1'] = binom.logpmf(k=new_blocks_adata.layers['alt'], n=new_blocks_adata.layers['total'], p = 0.5)
    new_blocks_adata.layers['p_cn2'] = binom.logpmf(k=new_blocks_adata.layers['alt'], n=new_blocks_adata.layers['total'], p = 1-epsilon)
    
    # remove columns with too few total counts
    valid_snvs = np.where(np.min(new_blocks_adata.layers['total'], axis = 0) >= min_snvcov_reads)[0]
    new_blocks_adata = new_blocks_adata[:, valid_snvs].copy()
    new_blocks_adata.obs = new_blocks_adata.obs.drop(columns = ['cluster_size'])
    
    return new_blocks_adata

wgd1_options = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    [2, 2],
]

wgd2_options = [
    [0, 0],
    [0, 1],
    [0, 2],
    [1, 0],
    [2, 0],
    [2, 2],
]


def compute_ll(partition_adata, return_ml_genotypes = False):
    ll_wgd1 = np.zeros((partition_adata.shape[1], len(wgd1_options)))
    for idx, (cn1, cn2) in enumerate(wgd1_options):
        ll_wgd1[:, idx] = partition_adata.layers[f'p_cn{cn1}'][0] + partition_adata.layers[f'p_cn{cn2}'][1]
    marginal_ll_wgd1 = scipy.special.logsumexp(ll_wgd1, axis=1)

    ll_wgd2 = np.zeros((partition_adata.shape[1], len(wgd2_options)))
    for idx, (cn1, cn2) in enumerate(wgd2_options):
        ll_wgd2[:, idx] = partition_adata.layers[f'p_cn{cn1}'][0] + partition_adata.layers[f'p_cn{cn2}'][1]
    marginal_ll_wgd2 = scipy.special.logsumexp(ll_wgd2, axis=1)
    
    if return_ml_genotypes:
        geno1 = [wgd1_options[a] for a in np.argmax(ll_wgd1, axis = 1)]
        geno2 = [wgd2_options[a] for a in np.argmax(ll_wgd2, axis = 1)]
        return marginal_ll_wgd1, marginal_ll_wgd2, geno1, geno2
    else:
        return marginal_ll_wgd1, marginal_ll_wgd2


def generate_null_resample(partition_adata, genotypes_1wgd, epsilon = 0.001, n_iter = 1000, return_values = False):
    P = np.clip(np.array(genotypes_1wgd).T / 2, a_min = epsilon, a_max = 1 - epsilon)
    n_snvs = partition_adata.shape[1]
    
    scores_resample = []
    if return_values:
        all_probs1 = []
        all_probs2 = []
    for i in range(n_iter):
        np.random.seed(i)

        # HACK: mask 0 entries with 1 and then deplete to get around binom.rvs issues
        sim_alts = binom.rvs(p = P, n = np.maximum(partition_adata.layers['total'], 1).astype(int))
        sim_alts = np.minimum(partition_adata.layers['total'], sim_alts)

        probs1 = np.zeros((3, n_snvs))
        probs1[0] = binom.logpmf(k=sim_alts[0], n=partition_adata.layers['total'][0], p = epsilon)
        probs1[1] = binom.logpmf(k=sim_alts[0], n=partition_adata.layers['total'][0], p = 0.5)
        probs1[2] = binom.logpmf(k=sim_alts[0], n=partition_adata.layers['total'][0], p = 1-epsilon)

        
        probs2 = np.zeros((3, n_snvs))
        probs2[0] = binom.logpmf(k=sim_alts[1], n=partition_adata.layers['total'][1], p = epsilon)
        probs2[1] = binom.logpmf(k=sim_alts[1], n=partition_adata.layers['total'][1], p = 0.5)
        probs2[2] = binom.logpmf(k=sim_alts[1], n=partition_adata.layers['total'][1], p = 1-epsilon)
        
        probs1, probs2 = compute_ll_numpy(probs1, probs2)
        scores_resample.append(probs2.sum() - probs1.sum())
        if return_values:
            all_probs1.append(probs1)
            all_probs2.append(probs2)
            
    if return_values:
        return scores_resample, np.array(all_probs1), np.array(all_probs2)
    else:
        return scores_resample


def compute_ll_numpy(probs1, probs2, return_ml_genotypes = False):
    assert np.array_equal(probs1.shape, probs2.shape)
    ll_wgd1 = np.zeros((probs1.shape[1], len(wgd1_options)))
    for idx, (cn1, cn2) in enumerate(wgd1_options):
        ll_wgd1[:, idx] = probs1[cn1] + probs2[cn2]
    marginal_ll_wgd1 = scipy.special.logsumexp(ll_wgd1, axis=1)

    ll_wgd2 = np.zeros((probs1.shape[1], len(wgd2_options)))
    for idx, (cn1, cn2) in enumerate(wgd2_options):
        ll_wgd2[:, idx] = probs1[cn1] + probs2[cn2]
    marginal_ll_wgd2 = scipy.special.logsumexp(ll_wgd2, axis=1)
    
    if return_ml_genotypes:
        geno1 = [wgd1_options[a] for a in np.argmax(ll_wgd1, axis = 1)]
        geno2 = [wgd2_options[a] for a in np.argmax(ll_wgd2, axis = 1)]
        return marginal_ll_wgd1, marginal_ll_wgd2, geno1, geno2
    else:
        return marginal_ll_wgd1, marginal_ll_wgd2


def enumerate_partitions(n, skip_reflection = True):
    if n == 2:
        yield np.array([1, 2])
    else:
        part = np.ones(n, dtype = int)
        a = np.arange(n)
        oddn = n if n % 2 == 1 else n - 1

        for k in range(1, int(n/2) + 1):
            last_j = int(math.factorial(n) / (math.factorial(k) * math.factorial(n - k))) - 1
            for j, idx in enumerate(itertools.combinations(a, k)):
                if skip_reflection:
                    if k > 1 and j == last_j:
                        continue
                my_part = part.copy()
                my_part[list(idx)] = 2
                yield my_part


def run_patient_partitions(sbm_adata, epsilon = 0.001, n_iter = 10000, min_snvcov_reads = 2):
    
    results = {}
    
    for block_column in ['sbmclone_cluster_id']:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
        blocks_adata = get_blocks_adata(sbm_adata, 'sbmclone_cluster_id')
        if blocks_adata.shape[1] == 0:
            print(f"Patient has no clonal cnLOH SNVs")
            break
        
        n_blocks = len(blocks_adata)
        for partition in enumerate_partitions(n_blocks):
            result = {}
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                partition_adata = get_partition(blocks_adata, partition, epsilon = epsilon, min_snvcov_reads = min_snvcov_reads)
            if partition_adata.shape[1] == 0:
                print(f"Patient has no clonal cnLOH SNVs with sufficient coverage for column {block_column} and partition {partition}")
                continue
            
            result['prob_1wgd'], result['prob_2wgd'], result['ml_geno_1wgd'], result['ml_geno_2wgd'] = compute_ll(partition_adata, return_ml_genotypes = True)
            result['score'] = result['prob_2wgd'].sum() - result['prob_1wgd'].sum()
            
            result['null_scores'] =  np.array(generate_null_resample(partition_adata, result['ml_geno_1wgd'], n_iter = n_iter))
            result['pvalue'] = np.sum(result['null_scores'] > result['score']) / n_iter
            result['partition_adata'] = partition_adata
            result['blocks_adata'] = blocks_adata
            results[block_column, tuple(partition)] = result
    return results


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
def main(adata_cna, adata_snv, tree_filename, output_cn, output_snv, output_pruned_tree, min_clone_size, min_num_snvs, min_prop_clonal_wgd, wgd_depth=0.5):
    adata = ad.read_h5ad(adata_cna)
    adata.obs['haploid_depth'] = adata.obs['coverage_depth'] / adata.obs['ploidy']

    snv_adata = ad.read_h5ad(adata_snv)

    # perform automatic WGD depth detection if wgd_depth is negative
    if wgd_depth < 0:
        print('Performing automatic WGD depth detection')
        # my_ad = get_blocks_adata(snv_adata, 'sbmclone_cluster_id')
        # part_ad = get_partition(my_ad, [1, 1, 2, 2])
        # print('part_ad', part_ad, sep='\n')
        # p1, p2, g1, g2 = compute_ll(part_ad, return_ml_genotypes = True)
        # n1 = generate_null_resample(part_ad, 1)
        # print('p2.sum() - p1.sum() = ', p2.sum() - p1.sum())
        # print('np.mean(n1) = ', np.mean(n1))
        # if p2.sum() > p1.sum():
        #     wgd_depth = 1  # set WGD depth to 1 if WGD2 has higher likelihood
        # else:
        #     wgd_depth = 0  # otherwise say that the WGD event is shared on the root clade

        # compute the likelihood of shared vs independent WGD events for every possible clone partition
        results = run_patient_partitions(snv_adata)

        print('results.keys()', results.keys(), sep='\n')

        # extract the most relevant results into a pandas dataframe with one row per partition
        rows = []
        for (col, partition), v in results.items():
            partition_adata = v['partition_adata']
            n_snvs = partition_adata.shape[1]
            size1, size2 = partition_adata.obs.partition_size
            rows.append((col, partition, size1, size2, size1 + size2, n_snvs, v['prob_1wgd'].sum(), v['prob_2wgd'].sum(), v['pvalue']))
        sbmdf = pd.DataFrame(rows, columns = ['cluster_column', 'partition', 'n_cells_p1', 'n_cells_p2', 'n_cells_total', 'n_snvs', 'll_1wgd', 'll_2wgd', 'pvalue'])
        # 'score' is the difference in log likelihood between shared and independent WGD events
        sbmdf['score'] = sbmdf.ll_2wgd - sbmdf.ll_1wgd

        print('sbmdf', sbmdf, sep='\n')

        # take the score from the highest scoring partition, this will tell is if any partition exists 
        # that supports having multiple independent WGD events
        best_score = sbmdf.iloc[sbmdf.score.argmax()]['score']

        # set the WGD depth based on the best score
        if best_score > 0:
            wgd_depth = 1  # multiple independent WGD events at one generation below the root
        else:
            wgd_depth = 0  # one shared WGD event at the root
        print('Automatic WGD depth detection has set wgd_depth to', wgd_depth)

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
    # the wgd_depth parameter controls whether a WGD event is added at the root of the tree
    # or multiple independent events are placed on branches `wgd_depth` generations below the root
    dt.tl.add_wgd_tree(tree, adata_cn_clusters, wgd_depth=wgd_depth)

    # split branches with a WGD event into two branches
    dt.tl.split_wgd_branches(tree)

    # Recursively add n_wgd to each clade
    dt.tl.count_wgd(tree.clade, 0)

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
