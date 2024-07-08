# doubleTime
doubleTime is a method to estimate the timing of whole-genome doubling event(s) on a clone tree using SNVs. The current doubleTree software is designed for DLP+ single-cell whole-genome sequencing data.

# Method

doubleTime consists of the following steps:

1. (optional) Construct clone tree from SBMClone output using the perfect phylogeny algorithm.
2. Construct clone copy-number profiles and summarize SNV-covering reads at the clone level.
3. Apply a statistical test to estimate the number of WGD events affecting the tree.
4. Assign SNVs to branches of the tree.
5. Correct branch lengths for variable SNV discovery in clones consisting of different numbers of cells.

# Dependencies

# Installation

# Usage
