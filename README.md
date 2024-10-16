# doubleTime
doubleTime is a method to estimate the timing of whole-genome doubling event(s) on a clone tree using SNVs. The current doubleTree software is designed for DLP+ single-cell whole-genome sequencing data.

## Method

doubleTime consists of the following steps:

1. Construct clone tree from SBMClone output using the perfect phylogeny algorithm.
2. Construct clone copy-number profiles and summarize SNV-covering reads at the clone level.
3. Assign SNVs to branches of the tree.

Coming soon:
* Apply a statistical test to estimate the number of WGD events affecting the tree.
* Correct branch lengths for variable SNV discovery in clones consisting of different numbers of cells.

## Inputs
* Single-cell haplotype-specific copy-number calls
* Counts for the number of SNV-covering and SNV-supporting reads in each cell for each SNV
* SBMClone results to group cells into clones
* Reference genome that was used for SNV and CNA analysis (FASTA file)

See `demo/input` for examples.

## Outputs
doubleTime produces output files with the following suffixes (prefix is the patient name):
* `_annotated_tree.pickle`: Clone tree with WGD events assigned to branches and SNV-derived branch lengths
* `_cna_clustered.h5` / `_snv_clustered.h5`: Clustered anndatas representing aggregate copy-number calls and SNV counts at the clone level
* `_tree_snv_assignment.csv`: Table containing SNV metrics and assignments to branches of the tree

An intermediate tree without annotations (`_tree.pickle`) is also produced and can be safely ignored or deleted.

See `demo/output` for examples.

# Setup

1. Clone this repository
2. Install dependencies from `environment.yml`: `conda env create -n doubletime --file environment.yml`

Optional: to run the demo, you will need to point `demo.yaml` to the reference genome `GRCh37-lite.fa`, which can be found here: https://www.bcgsc.ca/downloads/genomes/9606/hg19/1000genomes/bwa_ind/genome/GRCh37-lite.fa

# Usage

With the doubletime conda environment activated, we can execute the snakemake pipeline with the following command.
```
snakemake --snakefile doubleTime.smk --configfile demo.yaml --cores 1
```
Here, we specify the configuration file (`demo.yaml`), which contains the input data and parameters for the pipeline. We also specify the number of cores to use (1 in this case). Running doubleTree on the input data in `demo/input` should produce the output files in `demo/output`.
