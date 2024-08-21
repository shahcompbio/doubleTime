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
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
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

adata_filename = '/Users/parks17/Documents/repos/doubleTime/output/SPECTRUM-OV-036_general_clone_adata.h5'
tree_filename = '/Users/parks17/Documents/repos/doubleTime/output/SPECTRUM-OV-036_clones_pruned.pickle'
table_filename = '/Users/parks17/Documents/repos/doubleTime/output/SPECTRUM-OV-036_general_snv_tree_assignment.csv'

min_total_counts_perblock = 2
adata = ad.read_h5ad(adata_filename)
tree = pickle.load(open(tree_filename, 'rb'))
data = pd.read_csv(table_filename)

print(tree)
import doubletime.clonetreeplot
dir (doubletime.clonetreeplot)
```

```python
adata
```

```python
branches = [a.name for a in tree.find_clades()]
branch_length = data.query('clade != "none"').groupby(['clade']).size().reindex(branches, fill_value=0)
branch_length
```

```python
cluster_info = adata.obs
#cluster_info['branch_segment'] = {a.cluster_id: a.name for a in tree.get_terminals()}
cluster_info
```

```python
    # Set cluster id for leaves
    for clade in tree.find_clades():
        if clade.is_terminal():
            clade.cluster_id = clade.name.split('_')[1]
        else:
            clade.cluster_id = None
            
```

```python
fig = plt.figure()
ax = plt.gca()
doubletime.clonetreeplot.plot_clone_tree(tree, branch_length, cluster_info['cluster_size'], ax=ax)
```

```python
def annotate_wgd_timing(tree):
    """ Annotate each clade with the timing of the WGD event.

    Parameters
    ----------
    tree : Bio.Phylo.BaseTree.Tree
        Tree to annotate

    Returns
    -------
    Bio.Phylo.BaseTree.Tree
        Tree with WGD timing annotated
    """
    for clade in tree.find_clades():
        clade.wgd_timing = 'pre'
    for clade in tree.find_clades():
        if clade.is_wgd:
            for descendent in clade.find_clades():
                if descendent != clade:
                    descendent.wgd_timing = 'post'
    return tree
annotate_wgd_timing(tree)
```
