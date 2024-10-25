
import copy
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

n_wgd_colors = ['#CCCCCC', '#FC8D59', '#B30000']


def assign_plot_locations(tree):
    """
    Assign plotting locations to clades
    
    Parameters:
    - tree: A Bio.Phylo tree object
    
    Returns:
    - tree: A Bio.Phylo tree object with assigned values
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

    total_cell_count = sum([cell_counts[leaf.cluster_id] for leaf in tree.get_terminals()])
    for leaf in tree.get_terminals():
        leaf.cell_count = cell_counts[leaf.cluster_id]
        leaf.cell_fraction = cell_counts[leaf.cluster_id] / total_cell_count

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

    yticks = list(zip(*[(clade.branch_pos, f'{clade.cluster_id}, n={clade.cell_count}') for clade in tree.get_terminals()]))
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

