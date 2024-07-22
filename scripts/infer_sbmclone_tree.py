import anndata as ad
import numpy as np
from Bio.Phylo.BaseTree import Clade
import Bio.Phylo
from collections import defaultdict
import click
import logging
import sys
import pickle

def get_binary(adata, binarization_threshold = 0.01):
    density = np.zeros((len(adata.obs.sbmclone_cluster_id.unique()), len(adata.var.block_assignment.unique())))
    for rb, ridx in adata.obs.groupby('sbmclone_cluster_id'):
        for cb, cidx in adata.var.groupby('block_assignment'):
            density[int(rb), int(cb)] = np.minimum(adata[ridx.index, cidx.index].layers['alt'].todense(), 1).sum() / (len(ridx) * len(cidx))
    
    binary_density = density.copy()
    binary_density[density < binarization_threshold] = 0
    binary_density[density > binarization_threshold] = 1
    return density, binary_density


def construct_pp_tree(B):
    tuples = [tuple(c) for c in B.T]

    unique_cols, col_groups = np.unique(tuples, axis = 0, return_inverse = True)
    col_order = np.argsort(unique_cols.sum(axis = 1))[::-1]
    col2labels = defaultdict(lambda:list())
    for i in range(len(col_groups)):
        col2labels[col_order[col_groups[i]]].append(i)
    M = unique_cols.T[:, col_order].copy()
    if np.min(M.shape) <= 1:
        return construct_dummy_tree(*M.shape)

    pointers = [0]
    for j in range(1, M.shape[1]):
        prevs = []
        for i in range(M.shape[0]):
            if M[i, j] == 1:
                found0 = False
                for j2 in range(j-1, -1, -1):
                    if M[i, j2] == 1:
                        prevs.append(j2)
                        found0 = True
                        break  
        assert len(np.unique(prevs)) <= 1, (j, prevs, M)
        if len(prevs) > 0:
            pointers.append(prevs[0])
        else:
            pointers.append(-1)
    edges = {(pointers[i], i) for i in range(len(pointers)) if pointers[i] >= 0 and pointers[i] != i}

    # construct tree corresponding to characters
    adjacency_list = defaultdict(lambda:list())
    clades = {}
    for (u, v) in edges:
        if u not in clades:
            clades[u] = Clade(name=str(u))
            clades[u].mutations = "SNV_block_" + '/'.join([str(a) for a in col2labels[u]])
        if v not in clades:
            clades[v] = Clade(name=str(v))
            clades[v].mutations = "SNV_block_" + '/'.join([str(a) for a in col2labels[v]])
        adjacency_list[u].append(v)

    for u,vs in adjacency_list.items():
        clades[u].clades = [clades[v] for v in vs]

    T = Bio.Phylo.BaseTree.Tree(clades[0])
    T.clade.mutations = ''

    # assign clones to leaves, creating a leaf where necessary
    block2leaf = {}
    for i, r in enumerate(M):
        node = T.clade
        while len(node.clades) > 0:
            next_node = [a for a in node.clades if len(a.name) == 1 and r[int(a.name)] == 1]
            if len(next_node) == 0:
                # split off new leaf
                new_leaf = Clade(name=f'clone_{i}')
                new_leaf.mutations = ''
                node.clades.append(new_leaf)
                node = new_leaf
            else:
                assert len(next_node) == 1
                node = next_node[0]
        # reached leaf
        block2leaf[i] = node

    # relabel nodes
    leaf2blocks = {}
    for b, l in block2leaf.items():
        if l.name not in leaf2blocks:
            leaf2blocks[l.name] = []
        leaf2blocks[l.name].append(b)

    for leaf, clones in leaf2blocks.items():
        clades = [a for a in T.find_clades(leaf)]
        assert len(clades) == 1
        clade = clades[0]
        # label branch with relevant sbmclone column blocks
        clade.name = f'clone_{"/".join([str(c) for c in clones])}'

    for clade in T.find_clades():
        if not (clade.name.startswith('clone') or clade.name == 'root'):
            clade.name = 'internal_' + clade.name
        clade.branch_length = 1
    return T

def construct_dummy_tree(n_clones, n_snv_blocks):
    if n_clones == 1:
        # dummy tree is a 1-segment line with all SNVs on it
        my_clade = Clade(name='clone_0', branch_length = 1)
        my_clade.mutations = 'SNV_block_' + '/'.join([str(a) for a in np.arange(n_snv_blocks)])
        my_clade.is_wgd = False
        T = Bio.Phylo.BaseTree.Tree(my_clade)
        T.clade.mutations = ''
        T.clade.is_wgd = False
        return T
    elif n_snv_blocks == 1:
        # dummy tree is 1 branch and then a star
        clades = []
        for i in range(n_clones):
            my_clade = Clade(name=f'clone_{i}')
            my_clade.is_wgd = False
            clades.append(my_clade)
        T = Bio.Phylo.BaseTree.Tree(Clade(name='root', clades=clades, branch_length = 1))
        T.clade.mutations = 'SNV_block_0'
        T.clade.is_wgd = False
        print("found a patient with >1 cell block but ==1 SNV block")
        return T
    else:
        raise ValueError("Dummy tree should only be created if only 1 clone or SNV cluster is present")

@click.command()
@click.option('--snv_adata')
@click.option('--patient_id')
@click.option('--output')
def main(snv_adata, patient_id, output):
    snv_adata = ad.read_h5ad(snv_adata)

    Dens0, B0 = get_binary(snv_adata)
    colsums = np.sum(B0, axis = 0)
    empty_cols = np.where(colsums == 0)[0]
    non_empty_mask = np.ones(B0.shape[1], dtype=bool)
    non_empty_mask[empty_cols]=False
    B = B0[:, non_empty_mask]
    Dens = Dens0[:, non_empty_mask].copy()

    assert max(colsums) == B.shape[0]

    if min(B.shape) > 1:
        try:
            T = construct_pp_tree(B)
        except AssertionError as e:
            raise AssertionError(patient_id + ' violates perfect phylogeny: ' + str(Dens))

    else:
        T = construct_dummy_tree(*B.shape)

    with open(output, 'wb') as f:
        pickle.dump(T, f)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    main()
