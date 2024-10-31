import pandas as pd
import numpy as np
import Bio.Phylo


def add_wgd_tree(tree, adata_cn_clusters):
    '''
    For all branches in the tree, mark whether they contain a WGD event using the is_wgd parameter.
    This function currently assumes that all WGD events are shared rather than independent.
    
    Parameters
    ----------
    tree : Bio.Phylo.BaseTree.Tree
        Tree to add WGD events to
    adata_cn_clusters : anndata.AnnData
        Copy number data with n_wgd column in obs
    '''
    for clade in tree.find_clades():
        clade.is_wgd = False

    assert (adata_cn_clusters.obs['n_wgd'] == 0).all() or (adata_cn_clusters.obs['n_wgd'] == 1).all()
    tree.clade.is_wgd = (adata_cn_clusters.obs['n_wgd'] == 1).all()


def count_wgd(clade, n_wgd):
    '''
    Recursively count the number of WGD events in the tree
    
    Parameters
    ----------
    clade : Bio.Phylo.BaseTree.Clade
        Clade to count WGD events for
    n_wgd : int
        Number of WGD events in the root clade
    '''
    if clade.is_wgd:
        clade.n_wgd = n_wgd + 1
    else:
        clade.n_wgd = n_wgd
    for child in clade.clades:
        count_wgd(child, clade.n_wgd)


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
            post_wgd_clade.n_wgd = clade.n_wgd
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