# doubleTime
doubleTime is a method to estimate the timing of whole-genome doubling event(s) on a clone tree using SNVs. The current doubleTree software is designed for DLP+ single-cell whole-genome sequencing data.

## Method

doubleTime consists of the following steps:

1. (optional) Construct clone tree from SBMClone output using the perfect phylogeny algorithm.
2. Construct clone copy-number profiles and summarize SNV-covering reads at the clone level.
3. Apply a statistical test to estimate the number of WGD events affecting the tree. (not yet implemented)
4. Assign SNVs to branches of the tree.
5. Correct branch lengths for variable SNV discovery in clones consisting of different numbers of cells. (not yet implemented)

## Inputs
* Single-cell haplotype-specific copy-number calls
* Counts for the number of SNV-covering and SNV-supporting reads in each cell for each SNV
* SBMClone results to group cells into clones

## Outputs
* `_annotated_tree.pickle': Clone tree with WGD events assigned to branches and SNV-derived branch lengths
* `_cna_clustered.h5` / `_snv_clustered.h5`: Clustered anndatas representing aggregate copy-number calls and SNV counts at the clone level
* `_tree_snv_assignment.csv`: Table containing SNV metrics and assignments to branches of the tree

# Setup

1. Clone this repository
2. Install dependencies from `environment.yml`: `conda create -n doubletime --file environment.yml`

# Usage

```
snakemake --snakefile doubleTime.smk --configfile config.yaml
```
