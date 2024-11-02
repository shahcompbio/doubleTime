import pandas as pd
import numpy as np
import Bio.Phylo
import scgenome
import scipy
from scipy.stats import binom
import math
import itertools
import warnings


def add_wgd_tree(tree, adata_cn_clusters, wgd_depth=0):
    '''
    For all branches in the tree, mark whether they contain a WGD event using the is_wgd parameter.
    This function currently assumes that all WGD events are shared rather than independent.
    
    Parameters
    ----------
    tree : Bio.Phylo.BaseTree.Tree
        Tree to add WGD events to
    adata_cn_clusters : anndata.AnnData
        Copy number data with n_wgd column in obs
    wgd_depth : int
        The depth of the WGD event in the tree. This should be 0 if the WGD event is at the root of the tree,
        1 if the WGD event is on the children of the root, 2 if the WGD event is on the grandchildren of the root, etc.
    '''
    for clade in tree.find_clades():
        clade.is_wgd = False

    # make sure that all clones have either 0 or 1 WGD event
    assert (adata_cn_clusters.obs['n_wgd'] == 0).all() or (adata_cn_clusters.obs['n_wgd'] == 1).all()

    # if all clones have 1 WGD event, we know that there must be at least one WGD event in the tree
    if (adata_cn_clusters.obs['n_wgd'] == 1).all():
        # find all clades that are `wgd_depth` generations below the root
        current_clades = [tree.clade]
        for _ in range(wgd_depth):
            # find all the child clades that descend from the set of current clades
            child_clades = []
            for clade in current_clades:
               child_clades.extend(clade.clades)
            # set the child clades as the current clades
            current_clades = child_clades

        # set is_wgd to True for all the clades at the desired depth
        for clade in current_clades:
            clade.is_wgd = True



def count_wgd(clade, n_wgd=0):
    '''
    Recursively count the number of WGD events in the tree. For branches that were split into two by a WGD event, 
    the n_wgd parameter will be `n` before the WGD event and `n+1` after the WGD event.
    
    Parameters
    ----------
    clade : Bio.Phylo.BaseTree.Clade
        Clade to count WGD events for
    n_wgd : int
        Number of WGD events above the root clade, in most cases this should be 0.
    '''
    clade.n_wgd = n_wgd
    for child in clade.clades:
        if clade.is_wgd:
            count_wgd(child, n_wgd + 1)
        else:
            count_wgd(child, n_wgd)


def split_wgd_branches(tree):
    """ Split branches that are marked as WGD events into two branches, one
    before the WGD and one after.

    Parameters
    ----------
    tree : Bio.Phylo.BaseTree.Tree
        Tree to split
    
    Returns
    -------
    Bio.Phylo.BaseTree.Tree
        Tree with WGD branches split
    """
    for clade in list(tree.find_clades()):
        if clade.is_wgd:
            post_wgd_clade = Bio.Phylo.BaseTree.Clade(
                branch_length=1.,
                name='postwgd_' + clade.name,
                clades=clade.clades,
            )
            post_wgd_clade.is_wgd = False
            post_wgd_clade.wgd_timing = 'post'
            clade.clades = [post_wgd_clade]
            clade.wgd_timing = 'pre'
        else:
            clade.wgd_timing = 'pre'
    return tree


def build_cn_states_df(tree, cnA, cnB):
    '''
    Build a dataframe of all the possible CN states for each clone in the tree.
    This function currently assumes that there is at most one WGD event in the tree.
    Thus it only accounts for the following allele copy number states: 2|0, 2|1, 2|2, 1|1, 1|0.
    
    Parameters
    ----------
    tree : Bio.Phylo.BaseTree.Tree
        The tree of clades. Each clade should have a boolean attribute 'is_wgd' indicating whether there is a WGD event at this branch.
        There should be leafs attached to each clade. Each leaf should have an integer attribute 'n_wgd' indicating the number of WGD events and the name of the clone.
    cnA : int
        The copy number of the A allele.
    cnB : int
        The copy number of the B allele.

    Returns
    ----------
    cn_states_dfa : pd.DataFrame
        A dataframe of the possible CN states for the A allele based on the SNVs position within the tree.
    cn_states_dfb : pd.DataFrame
        A dataframe of the possible CN states for the B allele based on the SNVs position within the tree.
    '''

    def calculate_snv_multiplicity(cn, leaf_n_wgd, wgd_timing):
        if cn == 0:
            return 0
        elif wgd_timing == 'pre' and leaf_n_wgd == 1:
            return cn
        else:
            return 1

    cn_states_df = []
    for clade in tree.find_clades():
        assert clade.n_wgd <= 1
        for leaf in clade.get_terminals():
            cn_states_df.append({
                'clade': clade.name,
                'leaf': leaf.name,
                'cn_a': calculate_snv_multiplicity(cnA, leaf.n_wgd, clade.wgd_timing),
                'cn_b': calculate_snv_multiplicity(cnB, leaf.n_wgd, clade.wgd_timing)})

    # Add zero state to account for the case where the SNV is assigned outside the tree
    for leaf in tree.get_terminals():
        cn_states_df.append({
            'clade': 'none',
            'leaf': leaf.name,
            'cn_a': 0,
            'cn_b': 0})
        
    cn_states_dfa = pd.DataFrame(cn_states_df).set_index(['clade', 'leaf'])['cn_a'].unstack(fill_value=0)
    cn_states_dfb = pd.DataFrame(cn_states_df).set_index(['clade', 'leaf'])['cn_b'].unstack(fill_value=0)

    return cn_states_dfa, cn_states_dfb


def is_cpg_snv(data):
    """
    This function checks if a single nucleotide variant (SNV) is a C to T mutation
    in a CpG context or its reverse complement G to A in a CpG context.
    
    Parameters
    ----------
    data : pd.DataFrame
        A dataframe with columns 'ref', 'alt', and 'tri_nucleotide_context'
    
    Returns
    -------
    pd.Series
        A boolean series indicating whether the SNV is a C to T mutation in a CpG context
        or a G to A mutation in a CpG context on the reverse strand.
    """
    return np.logical_or(np.logical_and(data.ref == 'C', np.logical_and(data.alt == 'T', data.tri_nucleotide_context.str.slice(2) == 'G')),
                         np.logical_and(data.ref == 'G', np.logical_and(data.alt == 'A', data.tri_nucleotide_context.str.slice(0,1) == 'C')))


def is_apobec_snv(ref_base, alt_base, trinucleotide_context):
    """
    Classify a SNV as APOBEC-induced based on its substitution type and trinucleotide context.
    This function also accounts for the reverse complement context.
    
    Parameters
    ----------
    ref_base : str 
        The reference base (e.g., 'C').
    alt_base : str
        The alternate base (e.g., 'T').
    trinucleotide_context : str
        The trinucleotide context (e.g., 'TCA').
    
    Returns
    -------
    bool
        True if the SNV is APOBEC-induced, False otherwise.
    """

    # Check if the substitution is a C-to-T or G-to-A transition
    is_c_to_t = ref_base.upper() == 'C' and alt_base.upper() == 'T'
    is_g_to_a = ref_base.upper() == 'G' and alt_base.upper() == 'A'

    # Check if the substitution is a C-to-G or G-to-C transition
    is_c_to_g = ref_base.upper() == 'C' and alt_base.upper() == 'G'
    is_g_to_c = ref_base.upper() == 'G' and alt_base.upper() == 'C'

    # Check if the trinucleotide context fits the TpCpX pattern on the forward strand
    is_tpctx_forward = trinucleotide_context[1].upper() == 'C' and trinucleotide_context[0].upper() == 'T'

    # Check if the trinucleotide context fits the RpGpX pattern on the reverse strand (where R is A or G)
    is_tpctx_reverse = trinucleotide_context[1].upper() == 'G' and trinucleotide_context[2].upper() == 'A'

    # APOBEC-induced mutations are C-to-T or C-to-G in TpCpX context or reverse complement
    return ((is_c_to_t or is_c_to_g) and is_tpctx_forward) or ((is_g_to_a or is_g_to_c) and is_tpctx_reverse)


def is_c_to_t_in_cpg_context(ref_base, alt_base, trinucleotide_context):
    """
    This function checks if a single nucleotide variant (SNV) is a C to T mutation
    in a CpG context or its reverse complement G to A in a CpG context.
    
    Parameters
    ----------
    ref_base : str
        The reference nucleotide
    alt_base : str
        The alternate nucleotide
    trinucleotide_context : str
        The trinucleotide context of the SNV (string of 3 nucleotides)
    
    Returns
    ----------
    bool
        True if the mutation is a C to T mutation in a CpG context or a G to A mutation
        in a CpG context on the reverse strand, False otherwise.
    """
    
    # Check if the mutation is C to T in a CpG context on the forward strand
    if ref_base == 'C' and alt_base == 'T':
        if len(trinucleotide_context) == 3 and trinucleotide_context[1] == 'C' and trinucleotide_context[2] == 'G':
            return True

    # Check if the mutation is G to A in a CpG context on the reverse strand
    if ref_base == 'G' and alt_base == 'A':
        if len(trinucleotide_context) == 3 and trinucleotide_context[0] == 'C' and trinucleotide_context[1] == 'G':
            return True
    
    return False


def compute_clone_cell_counts(adata, tree):
    '''
    Find the number of cells assigned to each clone. If a clone is split into pre- and post-WGD clades, 
    the number of cells is assigned to the terminal post-WGD clade. This function is necessary prior to
    calling dt.pl.plot_clone_tree() and/or Bio.Phylo.draw(tree). 

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix containing the filtered doubleTree output.
    tree : Bio.Phylo.BaseTree.Tree
        Phylogenetic tree with branch lengths annotated by SNV counts.

    Returns
    -------
    cell_counts : pd.Series
        Number of cells assigned to each clone (i.e. terminal clade). Input for dt.pl.plot_clone_tree().
    '''
    # get all of the clade names in the tree that contain the keyword 'clone'
    # importantly, this includes the post-WGD clades for WGD branches split into pre- and post-WGD clades
    clone_names = [clade.name for clade in tree.find_clades() if 'clone' in clade.name]

    # find the number of cells assigned to each clone from doubleTree output
    cell_counts = adata.obs['cluster_size'].copy()

    # rename the index of cell_counts so that they match the clade names in the tree
    new_index = []
    for a in cell_counts.index:
        # find the elements of clone_names that end with the integer a
        # this should append `postwgd_clone_{a}`` if there is a postwgd clade, otherwise `clone_{a}`
        matching_clades = sorted([c for c in clone_names if c.endswith(str(a))])[::-1]
        new_index.append(matching_clades[0])
    cell_counts.index = new_index

    return cell_counts


def compute_branch_lengths(data, tree, cell_counts, CpG=False):
    '''
    Compute the branch lengths of the tree based on the number of SNVs in each clade. Branch lengths
    are stored both directly in the tree object and in a dictionary that maps clade names to branch
    lengths. This function is necessary prior to calling dt.pl.plot_clone_tree() and/or Bio.Phylo.draw(tree). 

    Parameters
    ----------
    data : pd.DataFrame
        doubleTree output table of SNVs with columns 'clade', 'snv_id', and (optionally) 'is_cpg'.
    tree : Bio.Phylo.BaseTree.Tree
        Phylogenetic tree with branch lengths annotated by SNV counts.
    cell_counts : pd.Series
        Number of cells assigned to each clone (i.e. terminal clade). Output of dt.tl.compute_clone_cell_counts().
    CpG : bool
        If True, compute branch lengths based on CpG SNVs only. Otherwise, use all SNVs.
        
    Returns
    -------
    branch_lengths : dict
        Dictionary mapping clade names to branch lengths. Input for plot_clone_tree().
    '''
    branch_lengths = {}
    for clade in tree.find_clades():
        # subset the doubleTree output table to only include SNVs in the current clade
        clade_df = data[data.clade == clade.name]
        # if we are only considering CpG SNVs, subset the table to only include CpG SNVs
        if CpG:
            clade_df = clade_df[clade_df.is_cpg]
        # compute the number of unique SNVs in the clade
        blen = len(clade_df.snv_id.unique())
        branch_lengths[clade.name] = blen
        clade.branch_length = blen
        # if this is a terminal clade, assign the correct cell count and fraction to the clade
        if clade.is_terminal():
            clade.cell_count = cell_counts.loc[clade.name]
            clade.cell_fraction = cell_counts.loc[clade.name] / cell_counts.sum()
    
    return branch_lengths


def compute_clade_apobec_fraction(data):
    """
    Compute the fraction of APOBEC-induced mutations in each clade. This is necessary input for
    dt.pl.plot_clone_tree() when plotting APOBEC fractions on the tree.
    
    Parameters
    ----------
    data : pd.DataFrame
        A dataframe with columns 'snv', 'clade', and 'is_apobec'.

    Returns
    -------
    pd.Series
        A series with the fraction of APOBEC-induced mutations in each clade.
    """
    return data[['snv', 'clade', 'is_apobec']].drop_duplicates().groupby(['clade'])['is_apobec'].mean()

############################################################################################################
## Functions to perform automatic detection of shared vs independent WGD events (wgd_depth = 0 or 1) ##
############################################################################################################

def get_blocks_adata(adata, cluster_col, epsilon = 0.001):
    '''
    Get the blocks_adata from the SNV adata object. The blocks_adata is a subset of the adata object
    that only contains the clonal SNVs (those with Maj=2 and Min=0 across all clones).

    Parameters
    ----------
    adata : anndata object
        anndata object with the SNV data. This is the output from sbmclone where rows are cells.
    cluster_col : str
        column name in adata.obs that contains the cluster id for each SNV
    epsilon : float
        error rate for the binomial distribution

    Returns
    -------
    blocks_adata : anndata object
        anndata object with the rows aggregated by the cluster_col. The rows correspond to the number of clones.
    '''
    # convert the original adata to blocks_adata by aggregating cells by the cluster_col
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
    # add vaf layer
    blocks_adata.layers['vaf'] = blocks_adata.layers['alt'] / (blocks_adata.layers['total'])
    # add layer for probablility of being CN state 0 (VAF near 0)
    blocks_adata.layers['p_cn0'] = binom.logpmf(k=blocks_adata.layers['alt'], n=blocks_adata.layers['total'], p = epsilon)
    # add layer for probablility of being CN state 1 (VAF near 0.5)
    blocks_adata.layers['p_cn1'] = binom.logpmf(k=blocks_adata.layers['alt'], n=blocks_adata.layers['total'], p = 0.5)
    # add layer for probablility of being CN state 2 (VAF near 1)
    blocks_adata.layers['p_cn2'] = binom.logpmf(k=blocks_adata.layers['alt'], n=blocks_adata.layers['total'], p = 1-epsilon)

    # subset blocks_adata to only the cnloh SNVs (those with Maj=2 and Min=0 across all clones)
    # only CNLOH SNVs should be used to infer the WGD depth
    blocks_adata.var['is_cnloh'] = np.all((blocks_adata.layers['Maj'] == 2) & (blocks_adata.layers['Min'] == 0), axis=0)
    blocks_adata = blocks_adata[:, blocks_adata.var['is_cnloh']]

    return blocks_adata


def get_partition(blocks_adata, partition, epsilon = 0.001, min_snvcov_reads = 2):
    '''
    Partition the blocks_adata into two partitions based on the partition list

    Parameters
    ----------
    blocks_adata : anndata object
        anndata object with the blocks data.
    partition : list
        list of integers with the partition of each block. Its length corresponds with the number of clones.
    epsilon : float
        error rate for the binomial distribution
    min_snvcov_reads : int
        minimum number of reads required for a block to be included in the analysis

    Returns
    -------
    new_blocks_adata : anndata object
        anndata object with the blocks data partitioned into two partitions
    '''

    assert len(partition) == len(blocks_adata), (len(partition), len(blocks_adata))
    partition = np.array(partition)
    part1_idx = np.where(partition == 1)[0]
    part2_idx = np.where(partition == 2)[0]
    assert len(part1_idx) > 0 and len(part2_idx) > 0, (part1_idx, part2_idx)
    
    new_blocks_adata = blocks_adata[:2].copy()

    # add layers for the clones belonging to the 1st partition
    new_blocks_adata.layers['alt'][0] = blocks_adata[part1_idx].layers['alt'].toarray().sum(axis = 0)
    new_blocks_adata.layers['ref'][0] = blocks_adata[part1_idx].layers['ref'].toarray().sum(axis = 0)
    new_blocks_adata.layers['total'][0] = blocks_adata[part1_idx].layers['total'].toarray().sum(axis = 0)
    new_blocks_adata.layers['Min'][0] = np.median(blocks_adata[part1_idx].layers['Min'], axis = 0)
    new_blocks_adata.layers['Maj'][0] = np.median(blocks_adata[part1_idx].layers['Maj'], axis = 0)
    new_blocks_adata.layers['state'][0] = np.median(blocks_adata[part1_idx].layers['state'], axis = 0)

    # add partition size and block names to the obs
    new_blocks_adata.obs['partition_size'] = [sum([blocks_adata.obs.cluster_size[i] for i in part1_idx]),
                                              sum([blocks_adata.obs.cluster_size[i] for i in part2_idx])]
    new_blocks_adata.obs['blocks'] = ['/'.join([blocks_adata.obs.iloc[a].name for a in l]) for l in [part1_idx, part2_idx]]
    
    # add layers for the clones belonging to the 2nd partition
    new_blocks_adata.layers['alt'][1] = blocks_adata[part2_idx].layers['alt'].toarray().sum(axis = 0)
    new_blocks_adata.layers['ref'][1] = blocks_adata[part2_idx].layers['ref'].toarray().sum(axis = 0)
    new_blocks_adata.layers['total'][1] = blocks_adata[part2_idx].layers['total'].toarray().sum(axis = 0)
    new_blocks_adata.layers['Min'][1] = np.median(blocks_adata[part2_idx].layers['Min'], axis = 0)
    new_blocks_adata.layers['Maj'][1] = np.median(blocks_adata[part2_idx].layers['Maj'], axis = 0)
    new_blocks_adata.layers['state'][1] = np.median(blocks_adata[part2_idx].layers['state'], axis = 0)
    
    # add vaf layer for both partitions
    new_blocks_adata.layers['vaf'] = new_blocks_adata.layers['alt'] / (new_blocks_adata.layers['total'])
    
    # add layers for marginal probabilities of CN states
    new_blocks_adata.layers['p_cn0'] = binom.logpmf(k=new_blocks_adata.layers['alt'], n=new_blocks_adata.layers['total'], p = epsilon)
    new_blocks_adata.layers['p_cn1'] = binom.logpmf(k=new_blocks_adata.layers['alt'], n=new_blocks_adata.layers['total'], p = 0.5)
    new_blocks_adata.layers['p_cn2'] = binom.logpmf(k=new_blocks_adata.layers['alt'], n=new_blocks_adata.layers['total'], p = 1-epsilon)
    
    # remove columns (SNVs) with too few total counts
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


def compute_ll(partition_adata, wgd1_options=wgd1_options, wgd2_options=wgd2_options, return_ml_genotypes=False):
    '''
    Compute the log-likelihood of the data under the shared and independent WGD models for the partition_adata.
    The function computes the log-likelihood of the data under the shared WGD model for every possible WGD genotype

    Parameters
    ----------
    partition_adata : anndata object
        anndata object with the blocks data for the partition. There should be layers for the probability of observing
        a given copy number state in the major and minor alleles for each block ('p_cn0', 'p_cn1', and 'p_cn2' layers).
    wgd1_options : list
        list with the possible WGD genotypes for the shared WGD model. This should be a list of lists with
        two elements (A and B ascn states). The default is [[0, 0], [0, 1], [1, 0], [1, 1], [2, 2]]
    wgd2_options : list
        list with the possible WGD genotypes for the independent WGD model. This should be a list of lists with
        two elements (A and B ascn states). The default is [[0, 0], [0, 1], [0, 2], [1, 0], [2, 0], [2, 2]]
    return_ml_genotypes : bool
        whether to return the most likely genotypes for the shared and independent WGD models

    Returns
    -------
    marginal_ll_wgd1 : np.array
        array with the log-likelihood of the data under the shared WGD model for every block
    marginal_ll_wgd2 : np.array
        array with the log-likelihood of the data under the independent WGD model for every block
    geno1 : list (optional)
        list with the most likely genotypes for the shared WGD model
    geno2 : list (optional)
        list with the most likely genotypes for the independent WGD model
    '''
    # compute the log-likelihood of the data under the shared WGD model for every possible WGD genotype
    ll_wgd1 = np.zeros((partition_adata.shape[1], len(wgd1_options)))
    for idx, (cn1, cn2) in enumerate(wgd1_options):
        ll_wgd1[:, idx] = partition_adata.layers[f'p_cn{cn1}'][0] + partition_adata.layers[f'p_cn{cn2}'][1]
    # compute the marginal log-likelihood by taking the log-sum-exp of the log-likelihoods along the genotypes axis
    marginal_ll_wgd1 = scipy.special.logsumexp(ll_wgd1, axis=1)

    # compute the log-likelihood of the data under the independent WGD model for every possible WGD genotype
    ll_wgd2 = np.zeros((partition_adata.shape[1], len(wgd2_options)))
    for idx, (cn1, cn2) in enumerate(wgd2_options):
        ll_wgd2[:, idx] = partition_adata.layers[f'p_cn{cn1}'][0] + partition_adata.layers[f'p_cn{cn2}'][1]
    # compute the marginal log-likelihood by taking the log-sum-exp of the log-likelihoods along the genotypes axis
    marginal_ll_wgd2 = scipy.special.logsumexp(ll_wgd2, axis=1)
    
    if return_ml_genotypes:
        # compute the most likely genotypes for the shared and independent WGD models if requested
        geno1 = [wgd1_options[a] for a in np.argmax(ll_wgd1, axis = 1)]
        geno2 = [wgd2_options[a] for a in np.argmax(ll_wgd2, axis = 1)]
        return marginal_ll_wgd1, marginal_ll_wgd2, geno1, geno2
    else:
        return marginal_ll_wgd1, marginal_ll_wgd2


def compute_ll_numpy(probs1, probs2, wgd1_options=wgd1_options, wgd2_options=wgd2_options, return_ml_genotypes=False):
    '''
    Compute the log-likelihood of the data under the shared and independent WGD models for the partition_adata.
    This is similar to compute_ll but uses numpy arrays as input instead of an anndata object.

    Parameters
    ----------
    probs1 : np.array
        array with the log-likelihood of a given copy number state being observed in the major allele.
        This is always of shape (3, n_snvs) cn1 and cn2 can only be 0, 1, or 2
    probs2 : np.array
        array with the log-likelihood of a given copy number state being observed in the minor allele.
        This is always of shape (3, n_snvs) cn1 and cn2 can only be 0, 1, or 2
    wgd1_options : list
        list with the possible WGD genotypes for the shared WGD model. This should be a list of lists with
        two elements (A and B ascn states). The default is [[0, 0], [0, 1], [1, 0], [1, 1], [2, 2]]
    wgd2_options : list
        list with the possible WGD genotypes for the independent WGD model. This should be a list of lists with
        two elements (A and B ascn states). The default is [[0, 0], [0, 1], [0, 2], [1, 0], [2, 0], [2, 2]]
    return_ml_genotypes : bool
        whether to return the most likely genotypes for the shared and independent WGD models

    Returns
    -------
    marginal_ll_wgd1 : np.array
        array with the log-likelihood of the data under the shared WGD model for every block
    marginal_ll_wgd2 : np.array
        array with the log-likelihood of the data under the independent WGD model for every block
    '''
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

def generate_null_resample(partition_adata, genotypes_1wgd, epsilon = 0.001, n_iter = 1000, return_values = False):
    '''
    Generate the null distribution of the log likelihood difference between the shared and independent WGD models
    by resampling the data under the shared WGD model. The function generates the null distribution by resampling
    the data under the shared WGD model and computing the log likelihood of the data under the shared and independent
    WGD models for each resampled dataset.

    Parameters
    ----------
    partition_adata : anndata object
        anndata object with the blocks data for the partition
    genotypes_1wgd : list
        list with the most likely genotypes for the shared WGD model
    epsilon : float
        error rate for the binomial distribution
    n_iter : int
        number of iterations for the null hypothesis resampling
    return_values : bool
        whether to return the log likelihood values for each resampled dataset

    Returns
    -------
    scores_resample : list
        list with the log likelihood difference between the shared and independent WGD models for each resampled dataset
    all_probs1 : np.array (optional)
        array with the log likelihood values under the shared WGD model for each resampled dataset
    all_probs2 : np.array (optional)
        array with the log likelihood values under the independent WGD model for each resampled dataset
    '''
    # compute the probability of the alternative allele for each block under the shared WGD model
    P = np.clip(np.array(genotypes_1wgd).T / 2, a_min = epsilon, a_max = 1 - epsilon)
    n_snvs = partition_adata.shape[1]
    
    scores_resample = []
    if return_values:
        all_probs1 = []
        all_probs2 = []
    
    # recompute the log likelihoods n_iter times
    for i in range(n_iter):
        np.random.seed(i)  # set a new seed for each iteration

        # simulate the number of alternative alleles for each block under the shared WGD model
        # HACK: mask 0 entries with 1 and then deplete to get around binom.rvs issues
        sim_alts = binom.rvs(p = P, n = np.maximum(partition_adata.layers['total'], 1).astype(int))
        sim_alts = np.minimum(partition_adata.layers['total'], sim_alts)

        # compute the probabily mass function for CN states of 0 (VAF=0), 1 (VAF=0.5), and 2 (VAF=1) 
        # for the major allele
        probs_cn1 = np.zeros((3, n_snvs))
        probs_cn1[0] = binom.logpmf(k=sim_alts[0], n=partition_adata.layers['total'][0], p = epsilon)
        probs_cn1[1] = binom.logpmf(k=sim_alts[0], n=partition_adata.layers['total'][0], p = 0.5)
        probs_cn1[2] = binom.logpmf(k=sim_alts[0], n=partition_adata.layers['total'][0], p = 1-epsilon)

        # compute the probabily mass function for CN states of 0 (VAF=0), 1 (VAF=0.5), and 2 (VAF=1)
        # for the minor allele
        probs2 = np.zeros((3, n_snvs))
        probs2[0] = binom.logpmf(k=sim_alts[1], n=partition_adata.layers['total'][1], p = epsilon)
        probs2[1] = binom.logpmf(k=sim_alts[1], n=partition_adata.layers['total'][1], p = 0.5)
        probs2[2] = binom.logpmf(k=sim_alts[1], n=partition_adata.layers['total'][1], p = 1-epsilon)
        
        # convert probability mass functions to log likelihoods of the shared vs independent WGD models
        probs1, probs2 = compute_ll_numpy(probs_cn1, probs2)
        # scores correspond to the difference in log likelihoods
        scores_resample.append(probs2.sum() - probs1.sum())
        if return_values:
            all_probs1.append(probs1)
            all_probs2.append(probs2)
            
    if return_values:
        return scores_resample, np.array(all_probs1), np.array(all_probs2)
    else:
        return scores_resample


def enumerate_partitions(n, skip_reflection = True):
    '''
    Enumerate all the possible partitions of n blocks into two sets. The function generates the partitions
    in a way that avoids generating the same partition twice. For example, if n = 3, the function will generate
    the following partitions: [1, 1, 2], [1, 2, 1], [2, 1, 1]. The function also allows to skip the reflection
    of the partitions, which is useful when the order of the partitions does not matter.

    Parameters
    ----------
    n : int
        number of blocks to partition
    skip_reflection : bool
        whether to skip the reflection of the partitions

    Yields
    -------
    partition : np.array
        array with the partition of each block
    '''
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


def run_partitions(snv_adata, epsilon = 0.001, n_iter = 10000, min_snvcov_reads = 2):
    '''
    Wrapper function to enumerate all the possible partitions of the blocks_adata and compute the log likelihood
    of the data under the shared and independent WGD models for each partition. The function also computes the
    p-value for the null hypothesis that the data is generated by a shared WGD event at the root of the clone tree.

    Parameters
    ----------
    snv_adata : anndata object
        anndata object with the SNV data. This is the output from sbmclone where rows are cells.
    epsilon : float
        error rate for the binomial distribution
    n_iter : int
        number of iterations for the null hypothesis resampling
    min_snvcov_reads : int
        minimum number of reads required for a block to be included in the analysis

    Returns
    -------
    results : dict
        dictionary with the results for each partition. The keys are tuples with the sbmclone cluster column name and the partition.
        The values are dictionaries with the following keys
        - prob_1wgd: log likelihood of the data under the shared WGD model
        - prob_2wgd: log likelihood of the data under the independent WGD model
        - ml_geno_1wgd: most likely genotypes for the shared WGD model
        - ml_geno_2wgd: most likely genotypes for the independent WGD model
        - score: difference in log likelihood between the shared and independent WGD models. Positive values indicate
                    that the data is more likely to be generated by the independent WGD model.
        - null_scores: array with the scores for the null hypothesis resampling
        - pvalue: p-value for the null hypothesis that the data is generated by a shared WGD event at the root of the clone tree.
                    Significant p-values indicate that the data strongly supports either shared or independent WGD.
        - partition_adata: anndata object with the blocks data for the partition
    '''
    
    results = {}
    
    for block_column in ['sbmclone_cluster_id']:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
        blocks_adata = get_blocks_adata(snv_adata, 'sbmclone_cluster_id')
        if blocks_adata.shape[1] == 0:
            print(f"Patient has no clonal cnLOH SNVs")
            break
        
        # number of blocks according to SBMclone
        n_blocks = len(blocks_adata)
        # iterate over all possible partitions containing n_blocks
        for partition in enumerate_partitions(n_blocks):
            result = {}
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # get the blocks_adata for the partition
                partition_adata = get_partition(blocks_adata, partition, epsilon = epsilon, min_snvcov_reads = min_snvcov_reads)
            if partition_adata.shape[1] == 0:
                print(f"Patient has no clonal cnLOH SNVs with sufficient coverage for column {block_column} and partition {partition}")
                continue
            
            # compute log likelihoods for the shared and independent WGD models
            result['prob_1wgd'], result['prob_2wgd'], result['ml_geno_1wgd'], result['ml_geno_2wgd'] = compute_ll(partition_adata, return_ml_genotypes = True)
            result['score'] = result['prob_2wgd'].sum() - result['prob_1wgd'].sum()
            
            # resample the null distribution and compare these scores to the observed score to compute a p-value
            result['null_scores'] =  np.array(generate_null_resample(partition_adata, result['ml_geno_1wgd'], n_iter = n_iter))
            result['pvalue'] = np.sum(result['null_scores'] > result['score']) / n_iter
            result['partition_adata'] = partition_adata
            result['blocks_adata'] = blocks_adata

            # store the results for the partition in a dictionary
            results[block_column, tuple(partition)] = result

    return results


def automatic_wgd_depth_detection(snv_adata):
    '''
    Automatically detect the WGD depth based on the SNV data. This function will return 1 if there is evidence
    for multiple independent WGD events at one generation below the root of the clone tree. Otherwise, it will return 0.
    The log likelihood of the data under the shared and independent WGD models is computed for every possible clone partition
    to determine the most likely WGD depth.

    Parameters
    ----------
    snv_adata : anndata object
        anndata object with the SNV data. This is the output from sbmclone where rows are cells.

    Returns
    -------
    wgd_depth : int
        WGD depth based on the SNV data. 0 means that the null hypothesis of one shared WGD event at the root is supported.
        1 means that there is evidence for multiple independent WGD events at one generation below the root. This output is
        used as input for dt.tl.add_wgd_tree().
    '''
    # compute the likelihood of shared vs independent WGD events for every possible clone partition
    results = run_partitions(snv_adata)

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

    # take the score from the highest scoring partition, this will tell is if any partition exists 
    # that supports having multiple independent WGD events
    best_row = sbmdf.iloc[sbmdf.score.argmax()]

    # set the WGD depth based on the best score and the p-value
    if best_row.score > 0 and best_row.pvalue < 1e-3:
        wgd_depth = 1  # multiple independent WGD events at one generation below the root
    else:
        wgd_depth = 0  # null hypothesis is one shared WGD event at the root
    
    return wgd_depth