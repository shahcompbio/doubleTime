import os

configfile: "config.yaml"

# input files
snv_adata = config['snv_adata']
cna_adata = config['cna_adata']

# scripts directory
repo_dir = config['repo_dir']
scripts_dir = os.path.join(repo_dir, 'scripts')

# output directory
outdir = config["outdir"]
outplotdir = os.path.join(outdir, "plots")
logdir = os.path.join(outdir, "logs")
patient_id = config["patient_id"]

# parameters for doubleTime algorithm
tree_snv_min_clone_size = config['tree_snv_min_clone_size']
tree_snv_min_num_snvs = config['tree_snv_min_num_snvs']
tree_snv_min_prop_clonal_wgd = config['tree_snv_min_prop_clonal_wgd']
genome_fasta_filename = config['genome_fasta_filename']

# parameters for binarization in SBMclone tree inference
if 'binarization_threshold' in config:
    binarization_threshold = config['binarization_threshold']
else:
    binarization_threshold = 0.01

# create output directories if they don't already exist
if not os.path.exists(outdir):
    os.makedirs(outdir)

if not os.path.exists(outplotdir):
    os.makedirs(outplotdir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


rule all:
    input: 
        os.path.join(outdir, f"{patient_id}_tree_snv_assignment.csv"),
        os.path.join(outplotdir, f"{patient_id}_wgd_tree.pdf"),
        os.path.join(outplotdir, f"{patient_id}_apobec_tree.pdf"),
        os.path.join(outplotdir, f"{patient_id}_snv_multiplicity.pdf")


rule infer_sbmclone_tree:
    input:
        snv_adata=snv_adata,
    params: 
        patient_id=patient_id,
        binarization_threshold=config['binarization_threshold'],
    resources:
        mem_mb=64000
    output:
        os.path.join(outdir, f"{patient_id}_tree.pickle")
    log:
        os.path.join(logdir, f"{patient_id}_infer_sbmclone_tree.log")
    shell:
        """
        python scripts/infer_sbmclone_tree.py --snv_adata {input.snv_adata} --patient_id {params.patient_id} --output {output} \
            --binarization_threshold {binarization_threshold} \
            &> {log}
        """

rule construct_clustered_snv_adata:
    input:
        cna_adata=cna_adata,
        snv_adata=snv_adata,
        tree=os.path.join(outdir, f"{patient_id}_tree.pickle"),
    params: 
        patient_id=patient_id,
    resources:
        mem_mb=128000
    output:
        clustered_snv_adata=os.path.join(outdir, f"{patient_id}_snv_clustered.h5"),
        clustered_cna_adata=os.path.join(outdir, f"{patient_id}_cna_clustered.h5"),
        pruned_tree=os.path.join(outdir, f"{patient_id}_annotated_tree.pickle"),
    log:
        os.path.join(logdir, f"{patient_id}_construct_clustered_snv_adata.log")
    shell:
        """
        python {scripts_dir}/construct_clustered_snv_adata.py \
            --adata_cna {input.cna_adata}  --adata_snv {input.snv_adata} --tree_filename {input.tree} \
            --min_clone_size {tree_snv_min_clone_size} --min_num_snvs {tree_snv_min_num_snvs} --min_prop_clonal_wgd {tree_snv_min_prop_clonal_wgd} \
            --output_cn {output.clustered_cna_adata} --output_snv {output.clustered_snv_adata} --output_pruned_tree {output.pruned_tree} \
            &> {log}
        """

rule assign_snvs_to_tree:
    input:
        tree=os.path.join(outdir, f"{patient_id}_annotated_tree.pickle"),
        adata=os.path.join(outdir, f"{patient_id}_snv_clustered.h5"),
    params: 
        patient_id=patient_id,
    output:
        table=os.path.join(outdir, f"{patient_id}_tree_snv_assignment.csv")
    log:
        os.path.join(logdir, f"{patient_id}_assign_snvs_to_tree.log")
    shell:
        """
        python {scripts_dir}/assign_snvs_to_tree.py --adata {input.adata} --tree {input.tree} --ref_genome {genome_fasta_filename} \
            --output {output.table} \
            &> {log}
        """

rule qc_output_plots:
    input:
        tree=os.path.join(outdir, f"{patient_id}_annotated_tree.pickle"),
        adata=os.path.join(outdir, f"{patient_id}_snv_clustered.h5"),
        table=os.path.join(outdir, f"{patient_id}_tree_snv_assignment.csv")
    params:
        patient_id=patient_id
    output:
        snv_reads_hist = os.path.join(outplotdir, f"{patient_id}_snv_reads_hist.pdf"),
        clone_hist = os.path.join(outplotdir, f"{patient_id}_clone_hist.pdf"),
        clone_pairwise_vaf = os.path.join(outplotdir, f"{patient_id}_clone_pairwise_vaf.pdf"),
        snv_multiplicity = os.path.join(outplotdir, f"{patient_id}_snv_multiplicity.pdf"),
        bio_phylo_tree = os.path.join(outplotdir, f"{patient_id}_bio_phylo_tree.pdf"),
        wgd_tree = os.path.join(outplotdir, f"{patient_id}_wgd_tree.pdf"),
        bio_phylo_cpg_tree = os.path.join(outplotdir, f"{patient_id}_bio_phylo_CpG_tree.pdf"),
        cpg_tree = os.path.join(outplotdir, f"{patient_id}_CpG_tree.pdf"),
        apobec_tree = os.path.join(outplotdir, f"{patient_id}_apobec_tree.pdf"),
    log:
        os.path.join(logdir, f"{patient_id}_qc_output_plots.log")
    shell:
        """
        python {scripts_dir}/plot_qc_output.py \
        --adata_filename {input.adata} --tree_filename {input.tree} --table_filename {input.table} --patient_id {params.patient_id} \
        -srh {output.snv_reads_hist} -ch {output.clone_hist} -cpv {output.clone_pairwise_vaf} \
        -sm {output.snv_multiplicity} -bpt {output.bio_phylo_tree} -wt {output.wgd_tree} \
        -bptc {output.bio_phylo_cpg_tree} -ct {output.cpg_tree} -at {output.apobec_tree} \
        &> {log}
        """
