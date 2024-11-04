import copy
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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


def draw_branch_wgd_apobec_fraction(ax, clade, apobec_fraction, bar_height=0.25):
    ''''
    Draw a red bars for the APOBEC fraction and grey bars for the non-APOBEC fraction of the given clade.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the bars on.
    clade : Bio.Phylo.BaseTree.Clade
        The clade to draw the bars for. 
        Contains the branch length, start, and position along with the name of the clade.
    apobec_fraction : dict
        A dictionary with clade.name as the key and the APOBEC fraction as the value.
    bar_height : float
        The height of the bars to draw.
    '''
    bars = []
    start = clade.branch_start
    length = clade.branch_length * apobec_fraction.get((clade.name), 0)
    bars.append({'start': start, 'length': length, 'color': 'r'})

    start += length
    length = clade.branch_length * (1. - apobec_fraction.get((clade.name), 0))
    bars.append({'start': start, 'length': length, 'color': '0.75'})

    for bar in bars:
        rect = patches.Rectangle(
            (bar['start'], clade.branch_pos-bar_height/2.), bar['length'], bar_height, 
            linewidth=0, edgecolor='none', facecolor=bar['color'])
        ax.add_patch(rect)


def draw_branch_wgd_fraction(ax, clade, bar_height=0.25):
    '''
    Draw a rectangluar bar for the WGD fraction of a clade. The bar is gray if happening before the WGD event,
    orange if happening after the WGD event.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the bars on.
    clade : Bio.Phylo.BaseTree.Clade
        The clade to draw the bars for.
    bar_height : float
        The height of the bars to draw.
    '''
    rect = patches.Rectangle(
        (clade.branch_start, clade.branch_pos-bar_height/2.), clade.branch_length, bar_height, 
        linewidth=0, edgecolor='none', facecolor=n_wgd_colors[clade.n_wgd])
    ax.add_patch(rect)


def draw_branch_wgd_event(ax, clade, bar_height=0.25):
    '''
    Draw a triangle at the top of the branch where the WGD event occurs.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the triangle on.
    clade : Bio.Phylo.BaseTree.Clade
        The clade to draw the triangle for.
    bar_height : float
        The height of the branch, used to determine the bottom of the triangle.
    '''
    if clade.is_wgd:
        ax.scatter([clade.branch_start + clade.branch_length], [clade.branch_pos+bar_height], marker='v', color='darkorange', linewidth=0)


def draw_branch_malignant_event(ax, clade, pre_malignant_interval, bar_height=0.25):
    '''
    Draw a blue triangle at the top of the branch where the malignant event occurs.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the triangle on.
    clade : Bio.Phylo.BaseTree.Clade
        The clade to draw the triangle for.
    pre_malignant_interval : float
        The fraction of the branch length at which the malignant event occurs.
    bar_height : float
        The height of the branch, used to determine the bottom of the triangle.
    '''
    ax.scatter([clade.branch_start + clade.branch_length * pre_malignant_interval], [clade.branch_pos+bar_height], marker='v', color='blue')


def draw_branch_links(ax, clade, bar_height=0.25):
    '''
    Draw vertical lines connecting the branches of the tree.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the lines on.
    clade : Bio.Phylo.BaseTree.Clade
        The clade to draw the lines for.
    bar_height : float
        The height of the lines to draw.
    '''
    if clade.is_terminal():
        return
    child_pos = [child.branch_pos for child in clade.clades]
    if len(child_pos) < 2:
        return
    ax.plot(
        [clade.branch_start + clade.branch_length, clade.branch_start + clade.branch_length],
        [min(child_pos)-bar_height/2., max(child_pos)+bar_height/2.], color='0.5', ls='-', linewidth=1)


def draw_leaf_tri_size(ax, clade, scale_cell_fraction=100):
    '''
    Draw a triangle at the end of the leaf branch representing the number of cells in the clone.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the triangle on.
    clade : Bio.Phylo.BaseTree.Clade
        The clade to draw the triangle for.
    scale_cell_fraction : int
        The scale factor for the size of the triangles representing the number of cells in each clone.
    '''
    # only draw the triangle for terminal branches (leaves) as these are clones with cell counts
    if clade.is_terminal():
        branch_end = clade.branch_start+clade.branch_length
        ax.scatter([branch_end], [clade.branch_pos], marker=4, c='k', s=clade.cell_fraction * scale_cell_fraction, linewidth=0)


def plot_clone_tree(tree, branch_lengths, cell_counts, apobec_fraction=None, pre_malignant_interval=None, ax=None, scale_cell_fraction=100):
    '''
    Plot a tree with branch lengths annotated by the number of SNVs and branch colors split by WGD status.
    Here, branches occuring before the WGD event are colored in gray, branches occuring after the WGD event are colored in orange.
    If abopec_fraction is not None, the tree branches are colored by the fraction of APOBEC SNVs in each clade instead of WGD status.
    
    Parameters
    ----------
    tree : Bio.Phylo.BaseTree.Tree
        The tree to plot.
    branch_lengths : dict
        A dictionary mapping clade names to branch lengths.
    cell_counts : pd.Series
        A series with the number of cells in each clade.
    apobec_fraction : pd.Series, optional
        A series with the fraction of APOBEC SNVs in each clade.
        If None, the tree branches are colored by WGD status. If not None, the tree branches are colored by APOBEC fraction.
    pre_malignant_interval : float, optional
        The fraction of the branch length at which the malignant event occurs.
    ax : matplotlib.axes.Axes, optional
        The axis to plot on. If None, a new figure is created.
    scale_cell_fraction : int, optional
        The scale factor for the size of the triangles representing the number of cells in each clone.

    Returns
    -------
    matplotlib.axes.Axes
        The axis with the plot.
    '''
    
    tree = copy.copy(tree)
    
    # assign branch lengths directly to the tree object
    for clade in tree.find_clades():
        clade.branch_length = branch_lengths[clade.name]

    # assign the number of cells to each clone (leaf objects)
    total_cell_count = sum([cell_counts[leaf.name] for leaf in tree.get_terminals()])
    for leaf in tree.get_terminals():
        leaf.cell_count = cell_counts[leaf.name]
        leaf.cell_fraction = cell_counts[leaf.name] / total_cell_count

    # compute the locations for each clade within the figure
    assign_plot_locations(tree)

    # create a new figure if ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 1), dpi=150)

    for clade in tree.find_clades():
        if apobec_fraction is not None:
            # draw the branch lengths with colors based on the APOBEC fraction
            draw_branch_wgd_apobec_fraction(ax, clade, apobec_fraction)
        else:
            # draw the branch lengths with colors based on the WGD status
            draw_branch_wgd_fraction(ax, clade)
        # draw an orange triangle at the top of each WGD event
        draw_branch_wgd_event(ax, clade)
        # draw the branch links
        draw_branch_links(ax, clade)
        # draw the number of cells in each clone as a triangle on the right side
        draw_leaf_tri_size(ax, clade, scale_cell_fraction=scale_cell_fraction)

    # add a blue triangle at the top of the branch where the malignant event occurs
    if pre_malignant_interval is not None:
        draw_branch_malignant_event(ax, tree.clade, pre_malignant_interval)

    # format the axes
    yticks = list(zip(*[(clade.branch_pos, f'{clade.name.replace("postwgd_", "").replace("_", " ")}, n={clade.cell_count}') for clade in tree.get_terminals()]))
    ax.set_yticks(*yticks)
    ax.yaxis.tick_right()
    sns.despine(ax=ax, trim=True, left=True, right=False)
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('right')
    ax.set_ylim((-0.5, tree.count_terminals() - 0.5))
    ax.set_xlabel('# SNVs')

    if apobec_fraction is not None:
        # add a legend for the APOBEC status
        legend_elements = [patches.Patch(color='r', label='APOBEC'),
                           patches.Patch(color='0.75', label='Other')]
        legend_1 = ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.4, 1), frameon=False, fontsize=8, title='Signature')
    else:
        # add a legend for the WGD status
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


def plot_snv_hist_facetgrid(data, col, row=None, x='vaf', hue='ascn', bins=20, binrange=(0, 1)):
    '''
    Create a FacetGrid of histograms according to the following input parameters. This is most
    commonly used to plot a histogram of the VAF for each clone.

    Parameters
    ----------
    data : pd.DataFrame
        A table where each row is a unique SNV with the following columns:
        - col: the column to facet by
        - row: the row to facet by
        - x: the variable to plot
        - hue: the variable to color by
    col : str
        The column to facet by
    row : str, optional
        The row to facet by
    x : str, optional
        The variable to plot along the x-axis within each histogram. This is typically the VAF.
    hue : str, optional
        The variable to color by. This is typically the ASCN.
    bins : int, optional
        The number of bins to use in the histogram.
    binrange : tuple, optional
        The range of the bins to use in the histogram.

    Returns
    ----------
    g : sns.FacetGrid
        A seaborn FacetGrid object containing the histograms
    '''
    if row is None:
        g = sns.FacetGrid(col=col, data=data, sharey=False, hue=hue)
    else:
        g = sns.FacetGrid(col=col, row=row, data=data, sharey=False, hue=hue)
    g.map_dataframe(sns.histplot, x=x, bins=bins, binrange=binrange)
    g.add_legend()
    return g