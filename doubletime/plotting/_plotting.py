
import copy
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import numpy as np
import seaborn as sns

n_wgd_colors = ['#CCCCCC', '#FC8D59', '#B30000']


def assign_plot_locations(tree):
    """
    Assign plotting locations to clades
    
    Parameters
    ----------
    tree: A Bio.Phylo tree object
    
    Returns
    ----------
    tree: A Bio.Phylo tree object with assigned values
    """

    def assign_branch_pos(clade, counter):
        """
        Recursive function to traverse the tree and assign values.
        """
        # Base case: if this is a leaf, assign the next value and return it
        if clade.is_terminal():
            clade.branch_pos = next(counter)
            return clade.branch_pos
        
        # Recursive case: assign the average of the child values
        child_values = [assign_branch_pos(child, counter) for child in clade]
        average_value = float(sum(child_values)) / float(len(child_values))
        clade.branch_pos = average_value
        return average_value
    
    assign_branch_pos(tree.clade, itertools.count())

    def assign_branch_start(clade, branch_start):
        """
        Recursive function to traverse the tree and assign values.
        """
        clade.branch_start = branch_start
        for child in clade:
            assign_branch_start(child, branch_start + clade.branch_length)
    
    assign_branch_start(tree.clade, 0)

    return tree


def compute_cell_counts(adata, tree):
    '''
    Find the number of cells assigned to each clone. If a clone is split into pre- and post-WGD clades, 
    the number of cells is assigned to the terminal post-WGD clade. This function is necessary prior to
    calling plot_clone_tree() and/or Bio.Phylo.draw(tree). 

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix containing the filtered doubleTree output.
    tree : Bio.Phylo.BaseTree.Tree
        Phylogenetic tree with branch lengths annotated by SNV counts.

    Returns
    -------
    cell_counts : pd.Series
        Number of cells assigned to each clone (i.e. terminal clade). Input for plot_clone_tree().
    '''
    # get all of the clade names in the tree
    # importantly, this includes the post-WGD clades for WGD branches split into pre- and post-WGD clades
    clade_names = [clade.name for clade in tree.find_clades()]

    # find the number of cells assigned to each clone from doubleTree output
    cell_counts = adata.obs['cluster_size'].copy()

    # rename the index of cell_counts so that they match the clade names in the tree
    new_index = []
    for a in cell_counts.index:
        # find the elements of clade_names that end with the integer a
        # this should append `postwgd_clone_{a}`` if there is a postwgd clade, otherwise `clone_{a}`
        matching_clades = sorted([c for c in clade_names if c.endswith(str(a))])[::-1]
        new_index.append(matching_clades[0])
    cell_counts.index = new_index

    return cell_counts


def compute_branch_lengths(data, tree, cell_counts, CpG=False):
    '''
    Compute the branch lengths of the tree based on the number of SNVs in each clade. Branch lengths
    are stored both directly in the tree object and in a dictionary that maps clade names to branch
    lengths. This function is necessary prior to calling plot_clone_tree() and/or Bio.Phylo.draw(tree). 

    Parameters
    ----------
    data : pd.DataFrame
        doubleTree output table of SNVs with columns 'clade', 'snv_id', and (optionally) 'is_cpg'.
    tree : Bio.Phylo.BaseTree.Tree
        Phylogenetic tree with branch lengths annotated by SNV counts.
    cell_counts : pd.Series
        Number of cells assigned to each clone (i.e. terminal clade). Output of compute_cell_counts().
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


def draw_branch_wgd_fraction(ax, clade, bar_height=0.25):
    if clade.wgd_timing == 'pre':
        rect = patches.Rectangle(
            (clade.branch_start, clade.branch_pos-bar_height/2.), clade.branch_length, bar_height, 
            linewidth=0, edgecolor='none', facecolor=n_wgd_colors[0])
        ax.add_patch(rect)

    else:
        assert clade.wgd_timing == 'post'
        rect = patches.Rectangle(
            (clade.branch_start, clade.branch_pos-bar_height/2.), clade.branch_length, bar_height, 
            linewidth=0, edgecolor='none', facecolor=n_wgd_colors[1])
        ax.add_patch(rect)


def draw_branch_wgd_event(ax, clade, bar_height=0.25):
    if clade.is_wgd:
        ax.scatter([clade.branch_start + clade.branch_length], [clade.branch_pos+bar_height], marker='v', color='darkorange', linewidth=0)


def draw_branch_malignant_event(ax, clade, pre_malignant_interval, bar_height=0.25):
    ax.scatter([clade.branch_start + clade.branch_length * pre_malignant_interval], [clade.branch_pos+bar_height], marker='v', color='blue')


def draw_branch_links(ax, clade, bar_height=0.25):
    if clade.is_terminal():
        return
    child_pos = [child.branch_pos for child in clade.clades]
    if len(child_pos) < 2:
        return
    ax.plot(
        [clade.branch_start + clade.branch_length, clade.branch_start + clade.branch_length],
        [min(child_pos)-bar_height/2., max(child_pos)+bar_height/2.], color='0.5', ls='-', linewidth=1)


def draw_leaf_tri_size(ax, clade, scale_cell_fraction=100):
    if clade.is_terminal():
        branch_end = clade.branch_start+clade.branch_length
        ax.scatter([branch_end], [clade.branch_pos], marker=4, c='k', s=clade.cell_fraction * scale_cell_fraction, linewidth=0)


def plot_clone_tree(tree, branch_lengths, cell_counts, pre_malignant_interval=None, ax=None, scale_cell_fraction=100):
    tree = copy.copy(tree)
    
    for clade in tree.find_clades():
        clade.branch_length = branch_lengths[clade.name]

    total_cell_count = sum([cell_counts[leaf.name] for leaf in tree.get_terminals()])
    for leaf in tree.get_terminals():
        leaf.cell_count = cell_counts[leaf.name]
        leaf.cell_fraction = cell_counts[leaf.name] / total_cell_count

    assign_plot_locations(tree)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 1), dpi=150)

    for clade in tree.find_clades():
        draw_branch_wgd_fraction(ax, clade)
        draw_branch_wgd_event(ax, clade)
        draw_branch_links(ax, clade)
        draw_leaf_tri_size(ax, clade, scale_cell_fraction=scale_cell_fraction)

    if pre_malignant_interval is not None:
        draw_branch_malignant_event(ax, tree.clade, pre_malignant_interval)

    yticks = list(zip(*[(clade.branch_pos, f'{clade.name}, n={clade.cell_count}') for clade in tree.get_terminals()]))
    ax.set_yticks(*yticks)
    ax.yaxis.tick_right()
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('right')
    ax.set_ylim((-0.5, tree.count_terminals() - 0.5))

    legend_elements = [patches.Patch(color=n_wgd_colors[0], label='0'),
                       patches.Patch(color=n_wgd_colors[1], label='1'),
                       patches.Patch(color=n_wgd_colors[2], label='2')]
    legend_1 = ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.4, 1), frameon=False, fontsize=8, title='#WGD')
    
    return ax


def generate_size_legend(value_size_dict, marker='o', color='blue', title=''):
    """
    Generates a legend showing different sizes of a specific marker for specific values.

    Parameters
    ----------
    value_size_dict : dict
        A dictionary where keys are the labels (values) and values are the sizes of the marker.
    marker : str
        The shape of the marker. Default is 'o' (circle).
    color : str
        Color of the markers. Default is 'blue'.
    title : str
        Text for legend title. Default ''.

    Returns
    -------
    legend : matplotlib.legend.Legend
        A matplotlib legend object.
    """
    # Create figure and axis
    fig, ax = plt.subplots(dpi=150)
    
    # Plot each marker size
    for label, size in value_size_dict.items():
        ax.scatter([], [], s=size, label=label, color=color, marker=marker, linewidth=0)
    
    # Create legend with the title
    legend = ax.legend(title=title, loc='upper left', labelspacing=0.5, frameon=False)
    
    # Hide the axes
    ax.axis('off')
    
    return {
        'fig': fig,
        'ax': ax,
        'legend': legend,
    }


def generate_clone_size_legend():
    """ Generates a clone size legend for a clone tree plot
    """
    value_size_dict = {
        '10%': 10,  # Size in points^2
        '40%': 40,
        '80%': 80,
    }

    return generate_size_legend(value_size_dict, marker='^', color='k', title='Clone Fraction')


def plot_snv_reads_hist(adata, ax=None, min_total_counts_perblock=2):
    """
    Plot a histogram of the minimum total counts of SNVs that pass the filter
    """
    adata.var['min_total_count'] = np.array(adata.layers['total_count'].min(axis=0))

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    else:
        fig = ax.get_figure()
    
    adata.var['min_total_count'].hist(bins=20, ax=ax)

    # compute the nubmer of SNVs that pass the filter in all clones
    n = (adata.var['min_total_count'] >= min_total_counts_perblock).sum()

    ax.set_xlabel("Minimum total reads")
    ax.set_ylabel("Count")
    ax.set_title(f"SNVs that pass filter\nn={n} SNVs with at least {min_total_counts_perblock} reads in all clones")

    return fig, ax


def plot_clone_hist(data):
    '''
    Plot a histogram for the total number of counts across clones.
    
    Parameters
    ----------
    data : pd.DataFrame
        A table with the following columns:
        - leaf: the name of the clone
        - total_counts: the total number of counts

    Returns
    ----------
    g : sns.FacetGrid
        A seaborn FacetGrid object containing the histograms. One histogram per clone.
    '''
    g = sns.FacetGrid(col='leaf', data=data, sharey=False)
    g.map_dataframe(sns.histplot, x='total_counts', bins=20, binrange=(0, 200))
    return g


def plot_clone_pairwise_vaf(data):
    '''
    Plot the pairwise VAF by clade/branch for clone pairs.

    Parameters
    ----------
    data : pd.DataFrame
        A table with the following columns:
        - snv: the SNV ID
        - clade: the clade (branch) name that the SNV belongs to
        - leaf: the clone name
        - vaf: the variant allele frequency
    
    Returns
    ----------
    g : sns.PairGrid
        A seaborn PairGrid object containing the pairwise VAF plots
    '''
    # Pairwise VAF by clade/branch for clone pairs
    plot_data = data.set_index(['snv', 'clade', 'leaf'])['vaf'].unstack().reset_index(level=1)
    plot_data['clade'] = plot_data['clade'].astype('category')
    g = sns.pairplot(data=plot_data, hue='clade')
    return g