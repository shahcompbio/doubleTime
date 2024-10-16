---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python
    language: python
    name: python
---

```python tags=["remove-cell"]
import os
import itertools
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import anndata as ad
import Bio.Phylo
import numpy as np
from copy import deepcopy
import wgs_analysis.snvs.mutsig
import sys
```
# Patient {{ patient_id }}


# Load tree and anndata


```python tags=["parameters"]

output_dir = '/data1/shahs3/users/myersm2/repos/warm-autopsy/data/doubletime'
patient_id = 'lx516'

adata_filename = os.path.join(output_dir, f'{patient_id}_snv_clustered.h5')
tree_filename = os.path.join(output_dir, f'{patient_id}_annotated_tree.pickle')
table_filename = os.path.join(output_dir, f'{patient_id}_tree_snv_assignment.csv')

min_total_counts_perblock = 2
```

```python
adata = ad.read_h5ad(adata_filename)
tree = pickle.load(open(tree_filename, 'rb'))
data = pd.read_csv(table_filename)

if np.min(adata.shape) == 0 or len(data) == 0:
    sys.exit(0)
```

```python
print(tree)
```

```python

adata.var[['snv_type', 'is_homogenous_cn']].value_counts()

```

```python tags=["hide-input"]

adata.var['min_total_count'] = np.array(adata.layers['total_count'].min(axis=0))

adata.var['min_total_count'].hist(bins=20)

print(f'Number of SNVs with at least {min_total_counts_perblock} reads in all clones:', 
      (adata.var['min_total_count'] >= min_total_counts_perblock).sum())

adata = adata[:, adata.var['min_total_count'] >= min_total_counts_perblock]

plt.xlabel("Minimum total reads")
plt.ylabel("Count")
plt.title("SNVs that pass filter")
```

```python tags=["hide-input"]

# filter adata based on tree

clones = []
for leaf in tree.get_terminals():
    clones.append(leaf.name.replace('clone_', ''))

adata = adata[clones].copy()
adata.obs

```

```python tags=["hide-input"]
snv_counts = data[['snv', 'clade', 'wgd_timing']].drop_duplicates().groupby(['clade', 'wgd_timing']).size()
snv_counts
```

# QC Plots




## Total counts across clones


```python tags=["hide-input"]

g = sns.FacetGrid(col='leaf', data=data, sharey=False)
g.map_dataframe(sns.histplot, x='total_counts', bins=20, binrange=(0, 200))

```


## Pairwise VAF by clade/branch for clone pairs


```python tags=["hide-input"]

plot_data = data.set_index(['snv', 'clade', 'leaf'])['vaf'].unstack().reset_index(level=1)
plot_data['clade'] = plot_data['clade'].astype('category')
sns.pairplot(data=plot_data, hue='clade')

```


## Pairwise VAF by cn state index for clone pairs

CN state index represents the clade/branch an SNV was assigned to along with whether the SNV is before or after any WGD on that branch


```python tags=["hide-input"]

plot_data = data.set_index(['snv', 'cn_state_idx', 'leaf'])['vaf'].unstack().reset_index(level=1)
plot_data['cn_state_idx'] = plot_data['cn_state_idx'].astype('category')
sns.pairplot(data=plot_data, hue='cn_state_idx')

```


## VAF for each SNV multiplicity


```python tags=["hide-input"]

g = sns.FacetGrid(col='cn_state_a', data=data, sharey=False, hue='ascn')
g.map_dataframe(sns.histplot, x='vaf', bins=20, binrange=(0, 1))
g.add_legend()

```


## VAF for each cn state index (row) and each clone (column)


```python tags=["hide-input"]

g = sns.FacetGrid(col='leaf', row='cn_state_idx', data=data, sharey=False, hue='ascn')
g.map_dataframe(sns.histplot, x='vaf', bins=20, binrange=(0, 1))
g.add_legend()

```


## VAF for the zero state


```python tags=["hide-input"]
zero_state_data = data.query('cn_state_a == 0 & cn_state_b == 0')
if len(zero_state_data) > 0:
    g = sns.FacetGrid(col='leaf', data=zero_state_data, sharey=False, hue='ascn')
    g.map_dataframe(sns.histplot, x='vaf', bins=20, binrange=(0, 1))
    g.add_legend()

```


## Pairwise VAF for pairs of clones for the zero state only


```python tags=["hide-input"]
if len(zero_state_data) > 0:
    plot_data = zero_state_data.set_index(['snv', 'clade', 'leaf'])['vaf'].unstack().reset_index(level=1)
    plot_data['clade'] = plot_data['clade'].astype('category')
    sns.pairplot(data=plot_data, hue='clade')

```


## Branch lengths as SNV counts (preferred)


```python
clone_sizes = adata.obs['cluster_size'].copy()
clone_sizes.index = [f'clone_{a}' for a in clone_sizes.index]
```

```python tags=["hide-input"]

for clade in tree.find_clades():
    clade_df = data[data.clade == clade.name]
    cntr = clade_df.wgd_timing.value_counts()
    
    clade.branch_length = len(clade_df.snv_id.unique())

    if 'prewgd' in cntr.index:
        assert 'postwgd' in cntr.index
        clade.wgd_fraction = 2 * cntr['prewgd'] / (2 * cntr['prewgd'] + cntr['postwgd'])
    if clade.is_terminal():
        clade.cell_count = clone_sizes.loc[clade.name]
        clade.cell_fraction = clone_sizes.loc[clade.name] / clone_sizes.sum()

Bio.Phylo.draw(tree)

```


# Plot the tree


```python
def count_wgd(clade, n_wgd):
    if clade.is_wgd:
        clade.n_wgd = n_wgd + 1
    else:
        clade.n_wgd = n_wgd
    for child in clade.clades:
        count_wgd(child, clade.n_wgd)

count_wgd(tree.clade, 0)

```

```python tags=["hide-cell"]

import itertools

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

assign_plot_locations(tree)

```

```python
snv_types = sorted(data.ascn.unique())
```

```python tags=["hide-input"]

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

n_wgd_colors = ['#CCCCCC', '#FC8D59', '#B30000']


def draw_branch_wgd(ax, clade, bar_height=0.25):
    if clade.is_wgd:
        length1 = clade.branch_length * clade.wgd_fraction
        length2 = clade.branch_length * (1. - clade.wgd_fraction)
        rect1 = patches.Rectangle((clade.branch_start, clade.branch_pos-bar_height/2.), length1, bar_height, 
                                  linewidth=0, edgecolor='none', facecolor=n_wgd_colors[clade.n_wgd-1])
        ax.add_patch(rect1)
        rect2 = patches.Rectangle((clade.branch_start + length1, clade.branch_pos-bar_height/2.), length2, bar_height, 
                                  linewidth=0, edgecolor='none', facecolor=n_wgd_colors[clade.n_wgd])
        ax.add_patch(rect2)
        ax.scatter([clade.branch_start + length1], [clade.branch_pos+bar_height], marker='v', color='darkorange')
    else:
        rect = patches.Rectangle(
            (clade.branch_start, clade.branch_pos-bar_height/2.), clade.branch_length, bar_height, 
            linewidth=0, edgecolor='none', facecolor=n_wgd_colors[clade.n_wgd])
        ax.add_patch(rect)

def draw_branch_links(ax, clade, bar_height=0.25):
    if not clade.is_terminal():
        child_pos = [child.branch_pos for child in clade.clades]
        plt.plot(
            [clade.branch_start + clade.branch_length, clade.branch_start + clade.branch_length],
            [min(child_pos)-bar_height/2., max(child_pos)+bar_height/2.], color='k', ls=':')

def draw_leaf_tri_size(ax, clade, bar_height=0.25, max_height=2.):
    if clade.is_terminal():
        expansion_height = bar_height + max(0.1, clade.cell_fraction) * (max_height - bar_height) # bar_height to 1.5

        # Transform to create a regular shaped triangle
        height = (ax.transData.transform([0, expansion_height]) - ax.transData.transform([0, 0]))[1]
        length = (ax.transData.inverted().transform([height, 0]) - ax.transData.inverted().transform([0, 0]))[0]

        branch_end = clade.branch_start+clade.branch_length
        branch_pos_bottom = clade.branch_pos-bar_height/2.
        branch_pos_top = clade.branch_pos+bar_height/2.

        vertices = [
            [branch_end, branch_pos_bottom],
            [branch_end, branch_pos_top],
            [branch_end + length, branch_pos_top + expansion_height / 2],
            [branch_end + length, branch_pos_bottom - expansion_height / 2],
        ]
        tri = patches.Polygon(vertices, linewidth=1, edgecolor='0.25', facecolor='0.25')
        ax.add_patch(tri)


fig, ax = plt.subplots(figsize=(5, 1), dpi=150)

for clade in tree.find_clades():
    draw_branch_wgd(ax, clade)
    draw_branch_links(ax, clade)

yticks = list(zip(*[(clade.branch_pos, f'{clade.name.replace("_", " ")}, n={clade.cell_count}') for clade in tree.get_terminals()]))
ax.set_yticks(*yticks)
ax.yaxis.tick_right()
sns.despine(trim=True, left=True, right=False)
ax.yaxis.tick_right()
ax.yaxis.set_ticks_position('right')
ax.set_xlabel('# SNVs')
ax.set_title(f'{patient_id}\n SNV types: {snv_types}', fontsize=10)


for clade in tree.find_clades():
    draw_leaf_tri_size(ax, clade, bar_height=0)


legend_elements = [patches.Patch(color=n_wgd_colors[0], label='0'),
                   patches.Patch(color=n_wgd_colors[1], label='1'),
                   patches.Patch(color=n_wgd_colors[2], label='2')]
legend_1 = ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.5, 1), frameon=False, fontsize=8, title='#WGD')
```

# look at APOBEC on the tree

```python
def is_apobec_snv(ref_base, alt_base, trinucleotide_context):
    """
    Classify a SNV as APOBEC-induced based on its substitution type and trinucleotide context.
    This function also accounts for the reverse complement context.

    Parameters:
    - ref_base: The reference base (e.g., 'C').
    - alt_base: The alternate base (e.g., 'T').
    - trinucleotide_context: The trinucleotide context (e.g., 'TCA').

    Returns:
    - True if the SNV is APOBEC-induced, False otherwise.
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
    
    Parameters:
    ref_base (str): The reference nucleotide
    alt_base (str): The alternate nucleotide
    trinucleotide_context (str): The trinucleotide context of the SNV (string of 3 nucleotides)
    
    Returns:
    bool: True if the mutation is a C to T mutation in a CpG context or a G to A mutation
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
```

```python tags=["hide-input"]
apobec_fraction = data[['snv', 'clade', 'wgd_timing', 'is_apobec']].drop_duplicates().groupby(['clade', 'wgd_timing'])['is_apobec'].mean()
apobec_counts = data[['snv', 'clade', 'wgd_timing', 'is_apobec']].drop_duplicates().groupby(['clade', 'wgd_timing', 'is_apobec']).size()

apobec_counts

```

```python tags=["hide-input"]

def draw_branch_wgd_fraction(ax, clade, bar_height=0.25):
    bars = []
    if clade.is_wgd:
        start = clade.branch_start
        length = clade.branch_length * clade.wgd_fraction
        bars.append({'start': start, 'length': length, 'color': n_wgd_colors[clade.n_wgd-1]})

        start += length
        length = clade.branch_length * (1. - clade.wgd_fraction)
        bars.append({'start': start, 'length': length, 'color': n_wgd_colors[clade.n_wgd]})

    else:
        bars.append({'start': clade.branch_start, 'length': clade.branch_length, 'color': n_wgd_colors[clade.n_wgd]})

    for bar in bars:
        rect = patches.Rectangle(
            (bar['start'], clade.branch_pos-bar_height/2.), bar['length'], bar_height, 
            linewidth=0, edgecolor='none', facecolor=bar['color'])
        ax.add_patch(rect)


def draw_branch_wgd_event(ax, clade, bar_height=0.25):
    if clade.is_wgd:
        length1 = clade.branch_length * clade.wgd_fraction
        ax.scatter([clade.branch_start + length1], [clade.branch_pos+bar_height], marker='v', color='darkorange')

        
def draw_branch_wgd_apobec_fraction(ax, clade, bar_height=0.25):
    bars = []
    if clade.is_wgd:
        start = clade.branch_start
        length = clade.branch_length * clade.wgd_fraction * apobec_fraction.get((clade.name, 'prewgd'), 0)
        bars.append({'start': start, 'length': length, 'color': 'r'})

        start += length
        length = clade.branch_length * clade.wgd_fraction * (1. - apobec_fraction.get((clade.name, 'prewgd'), 0))
        bars.append({'start': start, 'length': length, 'color': '0.75'})

        start += length
        length = clade.branch_length * (1. - clade.wgd_fraction) * apobec_fraction.get((clade.name, 'postwgd'), 0)
        bars.append({'start': start, 'length': length, 'color': 'r'})

        start += length
        length = clade.branch_length * (1. - clade.wgd_fraction) * (1. - apobec_fraction.get((clade.name, 'postwgd'), 0))
        bars.append({'start': start, 'length': length, 'color': '0.75'})

    else:
        start = clade.branch_start
        length = clade.branch_length * apobec_fraction.get((clade.name, 'none'), 0)
        bars.append({'start': start, 'length': length, 'color': 'r'})

        start += length
        length = clade.branch_length * (1. - apobec_fraction.get((clade.name, 'none'), 0))
        bars.append({'start': start, 'length': length, 'color': '0.75'})

    for bar in bars:
        rect = patches.Rectangle(
            (bar['start'], clade.branch_pos-bar_height/2.), bar['length'], bar_height, 
            linewidth=0, edgecolor='none', facecolor=bar['color'])
        ax.add_patch(rect)


def draw_apobec_fraction(ax, clade, bar_height=0.25):
    if clade.is_wgd:        
        length1 = clade.branch_length * clade.wgd_fraction
        length2 = clade.branch_length * (1. - clade.wgd_fraction)
        rect1 = patches.Rectangle((clade.branch_start, clade.branch_pos-bar_height/2.), length1, bar_height, 
                                  linewidth=0, edgecolor='none', facecolor=n_wgd_colors[clade.n_wgd-1])
        ax.add_patch(rect1)
        rect2 = patches.Rectangle((clade.branch_start + length1, clade.branch_pos-bar_height/2.), length2, bar_height, 
                                  linewidth=0, edgecolor='none', facecolor=n_wgd_colors[clade.n_wgd])
        ax.add_patch(rect2)
        ax.scatter([clade.branch_start + length1], [clade.branch_pos+bar_height], marker='v', color='darkorange')

    else:
        rect = patches.Rectangle(
            (clade.branch_start, clade.branch_pos-bar_height/2.), clade.branch_length, bar_height, 
            linewidth=0, edgecolor='none', facecolor=n_wgd_colors[clade.n_wgd])
        ax.add_patch(rect)

    length1 = clade.branch_length * clade.apobec_fraction
    length2 = clade.branch_length * (1. - clade.apobec_fraction)
    rect1 = patches.Rectangle((clade.branch_start, clade.branch_pos-bar_height/2.), length1, bar_height, 
                              linewidth=0, edgecolor='none', facecolor='r')
    ax.add_patch(rect1)
    rect2 = patches.Rectangle((clade.branch_start + length1, clade.branch_pos-bar_height/2.), length2, bar_height, 
                              linewidth=0, edgecolor='none', facecolor='0.75')
    ax.add_patch(rect2)


fig, ax = plt.subplots(figsize=(5, 1), dpi=150)

for clade in tree.find_clades():
    draw_branch_wgd_apobec_fraction(ax, clade)
    draw_branch_links(ax, clade)
    draw_branch_wgd_event(ax, clade)

yticks = list(zip(*[(clade.branch_pos, f'{clade.name.replace("_", " ")}, n={clade.cell_count}') for clade in tree.get_terminals()]))
ax.set_yticks(*yticks)
ax.yaxis.tick_right()
sns.despine(trim=True, left=True, right=False)
ax.yaxis.tick_right()
ax.yaxis.set_ticks_position('right')
ax.set_xlabel('# SNVs')

for clade in tree.find_clades():
    draw_leaf_tri_size(ax, clade, bar_height=0)

ax.set_title(f'Patient {patient_id} APOBEC')

```

# restrict to CpG SNVs

```python


cpg_tree = deepcopy(tree)
for clade in cpg_tree.find_clades():
    clade_df = data[(data.clade == clade.name) & (data.is_cpg)]
    
    clade.branch_length = len(clade_df.snv_id.unique())
    
    if clade.is_wgd:
        cntr = clade_df.wgd_timing.value_counts().reindex(['prewgd', 'postwgd'])
        clade.wgd_fraction = 2 * cntr['prewgd'] / (2 * cntr['prewgd'] + cntr['postwgd'])
    if clade.is_terminal():
        clade.cell_count = clone_sizes.loc[clade.name]
        clade.cell_fraction = clone_sizes.loc[clade.name] / clone_sizes.sum()


Bio.Phylo.draw(cpg_tree)
assign_plot_locations(cpg_tree)
```

```python
fig, ax = plt.subplots(figsize=(5, 1), dpi=150)

for clade in cpg_tree.find_clades():
    draw_branch_wgd(ax, clade)
    draw_branch_links(ax, clade)

yticks = list(zip(*[(clade.branch_pos, f'{clade.name.replace("_", " ")}, n={clade.cell_count}') for clade in tree.get_terminals()]))
ax.set_yticks(*yticks)
ax.yaxis.tick_right()
sns.despine(trim=True, left=True, right=False)
ax.yaxis.tick_right()
ax.yaxis.set_ticks_position('right')
ax.set_xlabel('# SNVs')
ax.set_title(f'{patient_id}\n SNV types: {snv_types}', fontsize=10)


for clade in cpg_tree.find_clades():
    draw_leaf_tri_size(ax, clade, bar_height=0)


legend_elements = [patches.Patch(color=n_wgd_colors[0], label='0'),
                   patches.Patch(color=n_wgd_colors[1], label='1'),
                   patches.Patch(color=n_wgd_colors[2], label='2')]
legend_1 = ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.5, 1), frameon=False, fontsize=8, title='#WGD')
```

```python

```
